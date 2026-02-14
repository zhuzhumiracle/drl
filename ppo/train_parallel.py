import os
import time
import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

# --- PPO 并行训练配置 ---
NUM_ENVS = 64
TOTAL_UPDATES = 2000
STEPS_PER_UPDATE = 256  # 每个环境采样步数
MINIBATCHES = 8
EPOCHS = 4
GAMMA = 0.99
LAMBDA = 0.95
CLIP_RANGE = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
LR = 2.5e-4
MAX_GRAD_NORM = 0.5
MODEL_PATH = "ppo_flappy_64core.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.actor = nn.Linear(128, act_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        h = self.shared(x)
        logits = self.actor(h)
        value = self.critic(h)
        return logits, value


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    advantages = np.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def train_parallel():
    envs = gym.vector.AsyncVectorEnv([
        lambda: gym.make("FlappyBird-v0", use_lidar=False) for _ in range(NUM_ENVS)
    ])

    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.n

    model = ActorCritic(obs_dim, act_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    obs, _ = envs.reset()
    best_score = 0

    pbar = tqdm(total=TOTAL_UPDATES, desc="PPO 并行训练")

    for update in range(TOTAL_UPDATES):
        # 存储 rollout
        obs_buf = []
        act_buf = []
        logp_buf = []
        rew_buf = []
        done_buf = []
        val_buf = []
        score_buf = np.zeros(NUM_ENVS, dtype=np.int32)

        for _ in range(STEPS_PER_UPDATE):
            obs_t = torch.FloatTensor(obs).to(DEVICE)
            with torch.no_grad():
                logits, values = model(obs_t)
                dist = Categorical(logits=logits)
                actions = dist.sample()
                logp = dist.log_prob(actions)

            next_obs, rewards, terms, truncs, infos = envs.step(actions.cpu().numpy())
            dones = np.logical_or(terms, truncs).astype(np.float32)

            if "score" in infos:
                try:
                    score_buf = np.array(infos["score"], dtype=np.int32)
                except Exception:
                    pass

            obs_buf.append(obs)
            act_buf.append(actions.cpu().numpy())
            logp_buf.append(logp.cpu().numpy())
            rew_buf.append(rewards)
            done_buf.append(dones)
            val_buf.append(values.cpu().numpy().squeeze(-1))

            obs = next_obs

            # 记录最高分
            if "final_info" in infos:
                for i in range(NUM_ENVS):
                    f_info = infos["final_info"][i]
                    if f_info and "score" in f_info:
                        if f_info["score"] > best_score:
                            best_score = f_info["score"]
                            torch.save(model.state_dict(), MODEL_PATH)

        # 计算优势和回报
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).to(DEVICE)
            _, last_values = model(obs_t)
        last_values = last_values.cpu().numpy().squeeze(-1)

        rewards = np.array(rew_buf)
        values = np.array(val_buf)
        dones = np.array(done_buf)

        advantages, returns = compute_gae(rewards, values, dones, last_values, GAMMA, LAMBDA)

        # 展平
        obs_arr = np.array(obs_buf).reshape(-1, obs_dim)
        act_arr = np.array(act_buf).reshape(-1)
        logp_arr = np.array(logp_buf).reshape(-1)
        adv_arr = advantages.reshape(-1)
        ret_arr = returns.reshape(-1)

        # 标准化优势
        adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

        # PPO 更新
        batch_size = obs_arr.shape[0]
        idxs = np.arange(batch_size)

        for _ in range(EPOCHS):
            np.random.shuffle(idxs)
            for start in range(0, batch_size, batch_size // MINIBATCHES):
                end = start + batch_size // MINIBATCHES
                mb_idx = idxs[start:end]

                mb_obs = torch.FloatTensor(obs_arr[mb_idx]).to(DEVICE)
                mb_act = torch.LongTensor(act_arr[mb_idx]).to(DEVICE)
                mb_logp_old = torch.FloatTensor(logp_arr[mb_idx]).to(DEVICE)
                mb_adv = torch.FloatTensor(adv_arr[mb_idx]).to(DEVICE)
                mb_ret = torch.FloatTensor(ret_arr[mb_idx]).to(DEVICE)

                logits, values = model(mb_obs)
                dist = Categorical(logits=logits)
                logp = dist.log_prob(mb_act)
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - mb_logp_old)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (mb_ret - values.squeeze(-1)).pow(2).mean()

                loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        pbar.set_postfix({"最高分": best_score})
        pbar.update(1)

    pbar.close()
    envs.close()
    print(f"训练完成，模型已保存到 {MODEL_PATH}")


if __name__ == "__main__":
    train_parallel()
