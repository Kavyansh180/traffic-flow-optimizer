# Use the official AWS Lambda Python 3.10 base image
FROM public.ecr.aws/lambda/python:3.10

# Copy the requirements file into the container
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install dependencies
# We use the CPU version of PyTorch to save space and ensure compatibility
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

# AWS Lambda bug fix: Swap to the headless version of OpenCV so it doesn't crash 
# looking for missing UI graphics drivers (libGL) on the serverless container.
RUN pip uninstall -y opencv-python opencv-python-headless && \
    pip install opencv-python-headless

# Copy the inference FastAPI code into the container
COPY inference/app.py ${LAMBDA_TASK_ROOT}/app.py
COPY ppo_traffic_model.zip /var/task/

# Set the CMD to your Mangum handler (this bridges AWS Lambda to FastAPI)
CMD [ "app.handler" ]