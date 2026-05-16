import os
import boto3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mangum import Mangum
from stable_baselines3 import PPO
import numpy as np

# Initialize FastAPI app
app = FastAPI(title="Traffic Flow Optimizer API")

# AWS S3 Configuration (These will be set via environment variables in Lambda)
S3_BUCKET = os.environ.get("S3_BUCKET_NAME", "traffic-flow-model-bucket")
MODEL_KEY = "ppo_traffic_model.zip"
# AWS Lambda only allows writing to the /tmp directory
LOCAL_MODEL_PATH = f"/tmp/{MODEL_KEY}" 

# Global variable to cache the loaded model between warm Lambda invocations
model = None

def load_model():
    """Downloads the model from S3 and loads it into memory."""
    global model
    if model is None:
        print("Initializing model...")
        
        # Check if model is already in /tmp (from a previous warm start)
        if not os.path.exists(LOCAL_MODEL_PATH):
            try:
                # Defaulting to Mumbai region as requested
                print(f"Downloading {MODEL_KEY} from S3 bucket {S3_BUCKET}...")
                s3 = boto3.client('s3', region_name='ap-south-1') 
                s3.download_file(S3_BUCKET, MODEL_KEY, LOCAL_MODEL_PATH)
                print("Download complete.")
            except Exception as e:
                print(f"Error downloading model from S3: {e}")
                # Fallback for local testing: look for the file in the root directory
                if os.path.exists(MODEL_KEY):
                    print(f"Found local model at {MODEL_KEY}, bypassing S3.")
                    model = PPO.load(MODEL_KEY)
                    return
                else:
                    raise RuntimeError("Could not find model locally or on S3.")
        
        # Load the model using stable-baselines3
        model = PPO.load(LOCAL_MODEL_PATH)
        print("Model loaded successfully.")

# Request Schema: Expects exactly 9 floats representing the environment observation
class ObservationState(BaseModel):
    # Format: [queue_0, queue_1, queue_2, queue_3, phase_0, phase_1, phase_2, phase_3, time_elapsed]
    observation: list[float]

@app.on_event("startup")
async def startup_event():
    """Trigger model load when the API starts up."""
    load_model()

@app.get("/")
def health_check():
    """Simple ping to check if the API is alive."""
    return {"status": "healthy", "message": "Traffic Flow Optimizer API is running."}

@app.post("/predict")
def predict_action(state: ObservationState):
    """
    Receives the current 9-value intersection state and returns the agent's chosen action (0-3).
    """
    global model
    if model is None:
        load_model()
        
    try:
        # Convert input list to numpy array for the model
        obs = np.array(state.observation, dtype=np.float32)
        
        # Predict the best action. deterministic=True removes randomness, giving the absolute best guess.
        action, _states = model.predict(obs, deterministic=True)
        
        # Convert numpy integer to a standard Python integer for JSON serialization
        return {"action": int(action)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Wrap the FastAPI app with Mangum to make it compatible with the AWS Lambda handler
handler = Mangum(app)