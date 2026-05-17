import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from market_env import GlostenMilgromEnv

if __name__ == "__main__":
    env = GlostenMilgromEnv(mu=0.3, gamma=0.05)

    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.001,
        gamma=0.99,
        ent_coef=0.01
    )

    mean_reward_before, std_reward_before = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Initial: {mean_reward_before:.2f} +/- {std_reward_before:.2f}")

    model.learn(total_timesteps=100_000)

    mean_reward_after, std_reward_after = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Final: {mean_reward_after:.2f} +/- {std_reward_after:.2f}")

    model.save("ppo_market_maker")