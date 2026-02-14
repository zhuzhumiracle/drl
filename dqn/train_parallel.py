import gymnasium as gym
import flappy_bird_gymnasium
import torch
import numpy as np
from tqdm import tqdm
from dqn_agent import DQNAgent

# --- 64核暴力配置 ---
NUM_ENVS = 64
BATCH_SIZE = 256        # 过大 Batch 会导致更新迟钝且不稳定
UPDATE_K = 4            # 过高 UTD 比例容易发散
TOTAL_STEPS = 300000    # 总步数
EPS_DECAY_STEPS = 150000 # 延长探索时间
TARGET_UPDATE_FREQ = 1000 # 目标网络过快同步会削弱稳定性
WARMUP_STEPS = 2000     # 先收集一段经验再开始学习

def train_parallel():
    print(f"检测到 64 核算力，开启高性能并行训练模式...")
    envs = gym.vector.AsyncVectorEnv([
        lambda: gym.make("FlappyBird-v0", use_lidar=False) for _ in range(NUM_ENVS)
    ])
    
    state_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.n
    agent = DQNAgent(state_dim, action_dim)
    
    states, _ = envs.reset()
    pbar = tqdm(total=TOTAL_STEPS, desc="复仇者训练模式")
    
    last_best_score = 0
    step_count = 0

    while step_count < TOTAL_STEPS:
        # 批量预测 (64个环境一齐上)
        actions = [agent.select_action(states[i]) for i in range(NUM_ENVS)]
        
        # 物理引擎并发推进
        next_states, rewards, terminateds, truncateds, infos = envs.step(actions)
        
        # 存储经验并捕获得分
        for i in range(NUM_ENVS):
            done = terminateds[i] or truncateds[i]
            real_next_state = next_states[i]
            
            # 处理终局状态
            if done and "final_observation" in infos:
                real_next_state = infos["final_observation"][i]
            
            # 存入错题本
            agent.memory.push(states[i], actions[i], rewards[i], real_next_state, done)
            
            # 记录最高分并存模型
            if done and "final_info" in infos:
                f_info = infos["final_info"][i]
                if f_info and "score" in f_info:
                    score = f_info["score"]
                    if score > last_best_score:
                        last_best_score = score
                        torch.save(agent.policy_net.state_dict(), "best_model_v2.pth")
        
        # --- 核心改进：压榨算力的高频学习 ---
        if len(agent.memory) > BATCH_SIZE and step_count > WARMUP_STEPS:
            for _ in range(UPDATE_K):
                agent.learn(BATCH_SIZE)
            
            # 线性衰减 Epsilon
            fraction = min(1.0, step_count / EPS_DECAY_STEPS)
            agent.epsilon = max(0.01, 1.0 + fraction * (0.01 - 1.0))
            
        states = next_states
        step_count += 1
        pbar.update(1)
        
        # 定期同步目标网络
        if step_count % TARGET_UPDATE_FREQ == 0:
            agent.sync_target_network()
            
        pbar.set_postfix({"最高": last_best_score, "探索率": f"{agent.epsilon:.2f}"})

    pbar.close()
    envs.close()
    print("复仇成功！模型已保存。")

if __name__ == "__main__":
    train_parallel()