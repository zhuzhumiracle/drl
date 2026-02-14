import multiprocessing

# 获取当前电脑的最大逻辑线程数
max_envs = multiprocessing.cpu_count()
print(f"你的电脑最多建议开启 {max_envs} 个并行环境！")