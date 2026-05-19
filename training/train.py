import os
import numpy as np
import mlflow
from stable_baselines3 import PPO
from env.intersection_env import TrafficIntersectionEnv as IntersectionEnv

def main():
    env = IntersectionEnv()
    
    mlflow.start_run()
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1
    )
    
    print("Starting PPO training for 200,000 timesteps...")
    model.learn(total_timesteps=200000)
    
    model_path = "ppo_traffic_model.zip"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    mlflow.end_run()
    
    # Validation Loop
    print("\n" + "="*50)
    print("RUNNING VALIDATION (5 Episodes)")
    print("="*50)
    
    val_episodes = 5
    total_throughput = 0
    total_queue_all_steps = []
    phase_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    total_steps_taken = 0
    
    for ep in range(val_episodes):
        obs, _ = env.reset()
        done = False
        ep_throughput = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Use action directly since new env doesn't override it
            actual_action = int(np.array(action).flatten()[0])
            phase_counts[actual_action] += 1
            
            ep_throughput += info["throughput"]
            total_queue_all_steps.append(float(info["total_queue"]))
            total_steps_taken += 1
            
            done = terminated or truncated
            
        total_throughput += ep_throughput
    
    avg_queue = np.mean(total_queue_all_steps)
    avg_throughput = total_throughput / val_episodes
    
    print(f"Average Total Queue per Step: {avg_queue:.2f} cars")
    print(f"Average Throughput per Episode: {avg_throughput:.2f} cars")
    print("\nPhase Selection Distribution:")
    
    for phase, count in phase_counts.items():
        pct = (count / total_steps_taken) * 100
        print(f"  Phase {phase}: {count} times ({pct:.1f}%)")
        
    print("="*50)

if __name__ == "__main__":
    main()