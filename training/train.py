import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mlflow
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# Import our custom environment
from env.intersection_env import TrafficIntersectionEnv

class MLflowCallback(BaseCallback):
    """
    Custom callback to log training metrics to MLflow safely.
    Logs every log_freq steps to avoid freezing the training loop with database writes.
    """
    def __init__(self, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.reward_buffer = []
        self.queue_buffer = []
        self.throughput_buffer = []

    def _on_step(self) -> bool:
        # Buffer the data
        if "rewards" in self.locals:
            self.reward_buffer.append(self.locals["rewards"][0])
        
        if "infos" in self.locals and len(self.locals["infos"]) > 0:
            info = self.locals["infos"][0]
            if "total_queue" in info:
                self.queue_buffer.append(info["total_queue"])
            if "throughput" in info:
                self.throughput_buffer.append(info["throughput"])

        # Only write to MLflow every self.log_freq steps
        if self.num_timesteps % self.log_freq == 0:
            if self.reward_buffer:
                mlflow.log_metric("avg_step_reward", np.mean(self.reward_buffer), step=self.num_timesteps)
            if self.queue_buffer:
                mlflow.log_metric("avg_total_queue", np.mean(self.queue_buffer), step=self.num_timesteps)
            if self.throughput_buffer:
                mlflow.log_metric("avg_throughput", np.mean(self.throughput_buffer), step=self.num_timesteps)
            
            # Clear buffers after logging
            self.reward_buffer = []
            self.queue_buffer = []
            self.throughput_buffer = []
                
        return True

def train():
    # 1. Set up MLflow tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Traffic_Flow_PPO_Optimization")

    with mlflow.start_run():
        print("Starting training run...")
        
        # 2. Instantiate and wrap the environment
        # The Monitor wrapper records episode statistics (reward, length) automatically
        env = TrafficIntersectionEnv()
        env = Monitor(env) 
        
        # Define and log hyperparameters
        hyperparams = {
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "n_steps": 1024,
            "batch_size": 64,
            "n_epochs": 10,
            "total_timesteps": 30000 
        }
        mlflow.log_params(hyperparams)

        # 3. Initialize the PPO model
        model = PPO(
            policy=hyperparams["policy"],
            env=env,
            learning_rate=hyperparams["learning_rate"],
            n_steps=hyperparams["n_steps"],
            batch_size=hyperparams["batch_size"],
            n_epochs=hyperparams["n_epochs"],
            verbose=1 
        )

        # 4. Train the agent with the fixed MLflow logger (logs every 1000 steps)
        mlflow_callback = MLflowCallback(log_freq=1000)
        model.learn(total_timesteps=hyperparams["total_timesteps"], callback=mlflow_callback)

        # 5. Save the trained model
        model_name = "ppo_traffic_model"
        model.save(model_name)
        print(f"Model saved locally as {model_name}.zip")
        
        # 6. Upload to MLflow
        mlflow.log_artifact(f"{model_name}.zip")
        
        print("Training complete! Run 'mlflow ui --backend-store-uri sqlite:///mlflow.db' to view logs.")

if __name__ == "__main__":
    if not os.path.exists("env"):
        print("Error: Please run this script from the root 'traffic-flow-optimizer' directory.")
        print("Command: python training/train.py")
        exit(1)
        
    train()