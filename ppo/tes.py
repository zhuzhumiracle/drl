import sys
import time
import gymnasium
import flappy_bird_gymnasium
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "ppo_flappy_64core.pth"
NUM_EVAL_EPISODES = 2
DEFAULT_FPS = 0


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


def run_episode(env, model, fps):
    state, info = env.reset()
    done = False
    score = 0
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits, _ = model(state_tensor)
            action = torch.argmax(logits, dim=-1).item()
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        score = info.get("score", score)
        if fps > 0:
            time.sleep(1 / fps)
    return score


def test():
    env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    model = ActorCritic(obs_dim, act_dim).to(DEVICE)
    model_path = sys.argv[1] if len(sys.argv) > 1 else MODEL_PATH
    fps = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_FPS

    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"成功加载模型: {model_path}")
    except FileNotFoundError:
        print(f"找不到模型文件：{model_path}。请先训练并保存模型。")
        return

    model.eval()

    print("开始演示...")
    scores = []
    for ep in range(NUM_EVAL_EPISODES):
        score = run_episode(env, model, fps)
        scores.append(score)
        print(f"第 {ep + 1} 局通过管子数: {score}")

    print(f"评估完成，平均分: {sum(scores)/len(scores):.2f}，最高分: {max(scores)}")
    env.close()


if __name__ == "__main__":
    test()
