import streamlit as st
import requests
import numpy as np

# Configure the page
st.set_page_config(page_title="RL Traffic Optimizer", layout="wide")

st.title("🚦 Autonomous Traffic Flow Optimizer")
st.write("Watch the Reinforcement Learning agent manage a 4-way intersection in real-time.")

# --- Sidebar Controls ---
st.sidebar.header("Simulation Settings")
# Default to the local FastAPI server we just tested
api_url = st.sidebar.text_input("FastAPI Endpoint", value="https://u7tivljb4txpkjnxhpzbydjlna0ynxka.lambda-url.eu-north-1.on.aws/predict")

# --- Initialize Simulation State in Streamlit ---
# We use st.session_state so the values persist across button clicks
if 'queues' not in st.session_state:
    st.session_state.queues = [15.0, 12.0, 4.0, 3.0]  # Initial traffic jams
    st.session_state.active_phase = 0
    st.session_state.time_elapsed = 0.0
    st.session_state.step_count = 0
    st.session_state.total_cleared = 0

if st.sidebar.button("Reset Simulation"):
    st.session_state.queues = [15.0, 12.0, 4.0, 3.0]
    st.session_state.active_phase = 0
    st.session_state.time_elapsed = 0.0
    st.session_state.step_count = 0
    st.session_state.total_cleared = 0
    st.rerun()

# Mapping the numeric phases to readable text
phase_names = {
    0: "🟢 North-South (Straight)",
    1: "🟢 East-West (Straight)",
    2: "🟢 North-South (Left Turn)",
    3: "🟢 East-West (Left Turn)"
}

# --- UI Dashboard ---
st.subheader(f"Step: {st.session_state.step_count} | Active Signal: **{phase_names[st.session_state.active_phase]}**")
st.write(f"⏱️ Time on current signal: {int(st.session_state.time_elapsed)} seconds")

# Display the 4 lanes using columns
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("North-South", f"{int(st.session_state.queues[0])} cars")
with col2:
    st.metric("East-West", f"{int(st.session_state.queues[1])} cars")
with col3:
    st.metric("N-S Left Turn", f"{int(st.session_state.queues[2])} cars")
with col4:
    st.metric("E-W Left Turn", f"{int(st.session_state.queues[3])} cars")

st.divider()

# --- Simulation Logic ---
# When the user clicks this button, we ask the AI what to do next
if st.button("Step Forward (Ask AI)", type="primary", use_container_width=True):
    
    # 1. Prepare the exact 9-float observation array our API expects
    phase_one_hot = [0.0, 0.0, 0.0, 0.0]
    phase_one_hot[st.session_state.active_phase] = 1.0
    
    observation = st.session_state.queues + phase_one_hot + [st.session_state.time_elapsed]
    
    # 2. Make the HTTP POST request to the FastAPI backend
    try:
        response = requests.post(api_url, json={"observation": observation})
        
        if response.status_code == 200:
            # The AI's chosen action
            action = response.json().get("action")
            
            # 3. Update Environment based on AI Action
            if action == st.session_state.active_phase:
                st.session_state.time_elapsed += 5.0 # Stayed green
            else:
                st.session_state.active_phase = action # Light changed
                st.session_state.time_elapsed = 0.0
            
            # Simulate cars driving through the green light
            cleared = min(st.session_state.queues[st.session_state.active_phase], 3.0)
            st.session_state.queues[st.session_state.active_phase] -= cleared
            st.session_state.total_cleared += cleared
            
            # Simulate new cars arriving randomly (Poisson distribution)
            arrivals = np.random.poisson([0.8, 0.8, 0.3, 0.3])
            for i in range(4):
                st.session_state.queues[i] += arrivals[i]
            
            # Increment the step counter and refresh the UI
            st.session_state.step_count += 1
            st.rerun()
            
        else:
            st.error(f"API Error: {response.text}")
            
    except Exception as e:
        st.error(f"Connection failed: {e}. Is your FastAPI server (uvicorn) running in the background?")

st.write(f"**Total cars successfully routed:** {int(st.session_state.total_cleared)}")