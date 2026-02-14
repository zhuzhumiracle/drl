import gymnasium
import flappy_bird_gymnasium
import torch
from dqn_agent import DQNAgent

def train():
    # 创建环境 (不渲染画面，训练速度极快)
    env = gymnasium.make("FlappyBird-v0", use_lidar=False)
    
    state_dim = env.observation_space.shape[0] # 12
    action_dim = env.action_space.n            # 2
    
    # 实例化智能体
    agent = DQNAgent(state_dim, action_dim)
    
    # 训练超参数
    EPISODES = 100000
    BATCH_SIZE = 64
    TARGET_UPDATE_FREQ = 10 # 每过 10 局同步一次目标网络
    
    best_score = 0
    
    print("开始训练...")
    
    for episode in range(EPISODES):
        state, info = env.reset()
        total_reward = 0
        score = 0 # 记录穿过的管子数
        done = False
        
        while not done:
            # 1. 选动作
            action = agent.select_action(state)
            
            # 2. 与环境交互
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 3. 存入经验池
            agent.memory.push(state, action, reward, next_state, done)
            
            # 4. 学习更新
            agent.learn(BATCH_SIZE)
            
            state = next_state
            total_reward += reward
            score = info.get('score', 0)
        
        # 衰减 epsilon
        agent.update_epsilon()
        
        # 定期同步目标网络
        if episode % TARGET_UPDATE_FREQ == 0:
            agent.sync_target_network()
            
       # 保存表现最好的模型
        if score > best_score:
            best_score = score
            torch.save(agent.policy_net.state_dict(), "best_flappy_bird.pth")
            print(f"*** 发现新纪录! 保存模型 - 局数: {episode}, 穿越管子数: {score} ***")
            
            # 【新增逻辑】：如果超过 500 分，直接认定为天下无敌，提前下课！
            if best_score >= 500:
                print("AI 已经超神，提前结束训练！")
                break
        # 每 10 局打印一次进度
        if episode % 1000 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward:.1f}, Score: {score}, Epsilon: {agent.epsilon:.3f}")
        
    env.close()
    print("训练结束！")

if __name__ == "__main__":
    train()