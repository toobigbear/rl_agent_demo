import random

# ===================== ① 出题模块：任务+验收规则 =====================
def generate_task():
    """出题：生成一个目标数字【训练题】；随机生成一个整数，范围包含 1 和 5，也就是只能产出：1,2,3,4,5 这五个数字中的某一个。"""
    target = random.randint(1,5)
    # 验收规则：agent猜的数字 == target 就算成功
    def reward_function(guess: int, target_num: int):
        if guess == target_num:
            return 10  # 猜对，正奖励
        else:
            return -1  # 猜错，负奖励
    return target, reward_function

# ===================== ② Rollout沙箱：单次完整尝试 =====================
def rollout(strategy, target_num, reward_func):
    """
    一次独立沙箱尝试（一次Rollout）
    strategy：Agent策略（各个数字选择概率）
    return：轨迹(动作) + 本次奖励
    """
    # 模型决定动作：按策略概率选择猜测数字
    action = random.choices(population=[1,2,3,4,5], weights=strategy, k=1)[0]
    # 环境执行，返回结果，计算奖励
    reward = reward_func(action, target_num)
    trajectory = [action] # 本次rollout轨迹：Agent做了什么动作
    return trajectory, reward

# ===================== ③ 评分模块：收集rollout，汇总奖励 =====================
# 本例直接在rollout拿到reward；复杂场景会执行完整多轮动作后统一打分

# ===================== ④ 更新模型（策略更新）=====================
def update_strategy(old_strategy, trajectories, rewards, lr=0.1):
    """
    策略更新：轨迹+奖励，更新选择概率
    lr：学习率，控制每次更新幅度
    """
    new_strategy = old_strategy.copy()
    for traj, r in zip(trajectories, rewards):
        action = traj[0]
        idx = action - 1
        if r > 0:
            # 拿到正奖励：增加这个动作的概率
            new_strategy[idx] += lr
        else:
            # 负奖励：降低动作概率
            new_strategy[idx] -= lr
    # 归一化：概率总和必须=1
    total = sum(new_strategy)
    for i in range(len(new_strategy)):
        new_strategy[i] = max(0.01, new_strategy[i]/total) # 防止概率为0完全不再探索
    return new_strategy

# ===================== ⑤ 换题验证：新任务测试模型 =====================
def test_on_new_task(strategy, test_rounds=10):
    """用全新题目评估当前Agent效果"""
    total_reward = 0
    for _ in range(test_rounds):
        target, reward_func = generate_task()
        traj, r = rollout(strategy, target, reward_func)
        total_reward += r
    avg_reward = total_reward / test_rounds
    print(f"\n==== ⑤换题验证【全新任务】，平均奖励={avg_reward:.2f}")
    return avg_reward

# ===================== 主训练循环，串联5步闭环 =====================
if __name__ == "__main__":
    # Agent初始策略：5个数字初始等概率，完全不懂，瞎猜
    agent_strategy = [0.2, 0.2, 0.2, 0.2, 0.2]
    train_episode = 20 # 训练多少道题
    rollout_per_task = 5 # 每一道题，做5次独立沙箱尝试rollout

    print("===== 开始Agent强化学习5步闭环演示 =====")
    for task_id in range(train_episode):
        # ①出题
        target_num, reward_fn = generate_task()
        print(f"\n【第{task_id+1}道训练题】目标数字={target_num}")

        all_trajectories = []
        all_rewards = []
        # ②多次尝试：每道题多次独立沙箱rollout
        for attempt in range(rollout_per_task):
            traj, rew = rollout(agent_strategy, target_num, reward_fn)
            all_trajectories.append(traj)
            all_rewards.append(rew)
            print(f"  Rollout{attempt+1}：猜测={traj[0]}，奖励={rew}")

        # ③评分：汇总所有rollout的轨迹与奖励
        task_avg_r = sum(all_rewards)/len(all_rewards)
        print(f"  本道题所有尝试平均奖励: {task_avg_r:.2f}")

        # ④更新模型（策略）
        agent_strategy = update_strategy(agent_strategy, all_trajectories, all_rewards)
        print(f"  更新后Agent策略(数字1~5概率)：{[round(p,3) for p in agent_strategy]}")

    # ⑤训练结束，全新题目验证模型
    test_on_new_task(agent_strategy, test_rounds=10)
