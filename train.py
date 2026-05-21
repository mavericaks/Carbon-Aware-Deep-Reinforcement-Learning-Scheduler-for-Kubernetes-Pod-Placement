import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from gym_env import CarbonAwareSchedEnv

def evaluate_baseline_least_utilized(env, steps=100):
    """
    Evaluates a heuristic baseline: schedule the pod on the node with
    the lowest CPU utilization. Never defers.
    """
    obs, info = env.reset()
    total_reward = 0.0
    total_carbon = 0.0
    total_sla_violations = 0
    total_overloads = 0

    for _ in range(steps):
        # Extract CPU utilization ratios of nodes (obs indices 0, 3, 6)
        cpu_ratios = [obs[0], obs[3], obs[6]]
        # Action is the node with the minimum CPU ratio
        action = int(np.argmin(cpu_ratios))
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_carbon += info.get("carbon_emission", 0.0)
        if info.get("forced_scheduling", False) or info.get("sla_violation", False):
            total_sla_violations += 1
        if info.get("overloaded", False):
            total_overloads += 1

        if terminated:
            break

    return total_reward, total_carbon, total_sla_violations, total_overloads

def evaluate_agent(model, env, steps=100):
    """
    Evaluates the trained DRL agent.
    """
    obs, info = env.reset()
    total_reward = 0.0
    total_carbon = 0.0
    total_sla_violations = 0
    total_overloads = 0
    total_deferred = 0

    for _ in range(steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        total_carbon += info.get("carbon_emission", 0.0)
        
        if info.get("forced_scheduling", False):
            total_sla_violations += 1
        if info.get("overloaded", False):
            total_overloads += 1
        if info.get("deferred", False):
            total_deferred += 1

        if terminated:
            break

    return total_reward, total_carbon, total_sla_violations, total_overloads, total_deferred

def main():
    # 1. Instantiate and check environment
    print("Checking Custom Environment...")
    env_check = CarbonAwareSchedEnv()
    check_env(env_check)
    print("Environment check passed!")

    # 2. Setup Vectorized Env for Training
    def make_env():
        return CarbonAwareSchedEnv()
    
    env = DummyVecEnv([make_env])

    # 3. Train DRL Agent using PPO
    print("Training DRL Agent (PPO)...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1
    )
    model.learn(total_timesteps=50000)

    # Save the model
    model.save("carbon_scheduler_model")
    print("Model saved to carbon_scheduler_model.zip")

    # 4. Evaluation and Comparison
    print("\nEvaluating and Comparing Schedulers...")
    eval_env = CarbonAwareSchedEnv(max_steps=100)
    
    # Run Baseline
    base_reward, base_carbon, base_sla, base_overload = evaluate_baseline_least_utilized(eval_env, steps=100)
    
    # Run DRL Agent
    agent_reward, agent_carbon, agent_sla, agent_overload, agent_deferred = evaluate_agent(model, eval_env, steps=100)

    print("=" * 50)
    print(f"{'Metric':<25} | {'Least-Utilized (Base)':<22} | {'DRL Agent (PPO)':<15}")
    print("-" * 50)
    print(f"{'Total Reward':<25} | {base_reward:<22.2f} | {agent_reward:<15.2f}")
    print(f"{'Total Carbon (gCO2eq)':<25} | {base_carbon:<22.2f} | {agent_carbon:<15.2f}")
    print(f"{'SLA Violations':<25} | {base_sla:<22} | {agent_sla:<15}")
    print(f"{'Node Overloads':<25} | {base_overload:<22} | {agent_overload:<15}")
    print(f"{'Workloads Deferred':<25} | {'0 (Heuristic)':<22} | {agent_deferred:<15}")
    print("=" * 50)

if __name__ == "__main__":
    main()
