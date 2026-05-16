import numpy as np
import os
import sys

# Ensure we can import the environment if running from the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.intersection_env import TrafficIntersectionEnv

def evaluate_baseline(episodes=5):
    """
    Evaluates a fixed-timer traffic controller.
    Cycles through all 4 phases equally, assigning 30 seconds to each.
    """
    env = TrafficIntersectionEnv()
    
    # 30 seconds per phase / 5 seconds per timestep = 6 timesteps per phase
    steps_per_phase = 6 
    
    all_episode_metrics = []

    print(f"Starting Fixed-Timer Baseline Evaluation for {episodes} episodes...\n")

    for ep in range(episodes):
        obs, _ = env.reset()
        terminated, truncated = False, False
        
        step_count = 0
        ep_throughput = 0
        ep_max_queue = 0
        ep_total_queue_sum = 0 
        
        while not (terminated or truncated):
            # Fixed timer logic: 
            # integer division by 6 gives us the phase index, modulo 4 loops it back to 0
            # Result: 0,0,0,0,0,0 -> 1,1,1,1,1,1 -> 2,2,2,2,2,2 -> 3,3,3,3,3,3 -> repeat
            action = (step_count // steps_per_phase) % 4
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Accumulate metrics for comparison
            ep_throughput += info["throughput"]
            ep_max_queue = max(ep_max_queue, info["max_queue"])
            ep_total_queue_sum += info["total_queue"]
            
            step_count += 1
            
        # Average wait time proxy: average number of cars waiting across all steps
        avg_queue_per_step = ep_total_queue_sum / step_count
        
        all_episode_metrics.append({
            "throughput": ep_throughput,
            "max_queue": ep_max_queue,
            "avg_queue": avg_queue_per_step
        })
        
        print(f"Episode {ep+1} Results:")
        print(f"  Total Throughput: {ep_throughput:.0f} cars cleared")
        print(f"  Max Queue Reached: {ep_max_queue:.0f} cars in a single lane")
        print(f"  Avg Queue Length: {avg_queue_per_step:.2f} cars waiting per step\n")

    # Calculate final averages across all test episodes
    avg_throughput = np.mean([m["throughput"] for m in all_episode_metrics])
    avg_max_q = np.mean([m["max_queue"] for m in all_episode_metrics])
    avg_q = np.mean([m["avg_queue"] for m in all_episode_metrics])
    
    print("=========================================")
    print("   BASELINE PERFORMANCE (OVERALL AVG)    ")
    print("=========================================")
    print(f"Avg Total Throughput : {avg_throughput:.1f} cars")
    print(f"Avg Max Queue Peak   : {avg_max_q:.1f} cars")
    print(f"Avg Waiting Cars/Step: {avg_q:.2f} cars")
    print("=========================================")

if __name__ == "__main__":
    if not os.path.exists("env"):
        print("Error: Please run this script from the root 'traffic-flow-optimizer' directory.")
        print("Command: python baseline/fixed_timer.py")
        exit(1)
        
    evaluate_baseline()