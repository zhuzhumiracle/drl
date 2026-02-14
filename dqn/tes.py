import gymnasium
import flappy_bird_gymnasium
import torch
import time
import sys
from dqn_agent import QNetwork

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "best_model_v2.pth"
NUM_EVAL_EPISODES = 10

def run_episode(env, model):
    state, info = env.reset()
    done = False
    score = 0
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state_tensor)
            action = q_values.argmax().item()
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        score = info.get("score", score)
        time.sleep(1 / 30)
    return score

def test():
    # 创建带画面的环境
    env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 初始化网络并加载训练好的权重
    model = QNetwork(state_dim, action_dim).to(device)
    model_path = sys.argv[1] if len(sys.argv) > 1 else MODEL_PATH
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功加载模型: {model_path}")
    except FileNotFoundError:
        print(f"找不到模型文件：{model_path}。请先训练并保存模型。")
        return

    model.eval() # 设置为评估模式
    
    print("开始演示...")
    scores = []
    for ep in range(NUM_EVAL_EPISODES):
        score = run_episode(env, model)
        scores.append(score)
        print(f"第 {ep + 1} 局通过管子数: {score}")

    print(f"评估完成，平均分: {sum(scores)/len(scores):.2f}，最高分: {max(scores)}")
    env.close()

if __name__ == "__main__":
    test()