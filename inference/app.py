from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mangum import Mangum
from stable_baselines3 import PPO
import numpy as np

app = FastAPI(title="Traffic Flow Optimizer API")

MODEL_PATH = "ppo_traffic_model.zip"

model = None

def load_model():
    global model
    if model is None:
        print("Initializing model...")
        try:
            model = PPO.load(MODEL_PATH)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise RuntimeError(f"Could not load model from {MODEL_PATH}")

# Updated schema — removed time_since_green, matches new 9-float obs space
class TrafficState(BaseModel):
    queues: list[float]
    current_phase: int = 0
    time_on_phase: int = 0

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "Traffic Flow Optimizer API is running."}

@app.post("/predict")
def predict_action(state: TrafficState):
    global model
    if model is None:
        load_model()
        
    try:
        # Build 9-float observation matching TrafficIntersectionEnv._get_obs()
        # [queue_0, queue_1, queue_2, queue_3, phase_one_hot(4), time_since_change]
        
        # 1. Raw queue lengths (no normalization — env uses raw values)
        queues = list(state.queues)
        
        # 2. One-hot encode current phase
        one_hot = [1.0 if i == state.current_phase else 0.0 for i in range(4)]
        
        # 3. Time since change in seconds (steps × 5 seconds per step)
        time_elapsed = [float(state.time_on_phase * 5.0)]
        
        # Final 9-float observation
        obs = np.array(queues + one_hot + time_elapsed, dtype=np.float32)
        
        action, _ = model.predict(obs, deterministic=True)
        
        return {"phase": int(action)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

handler = Mangum(app)