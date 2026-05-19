import streamlit as st
import requests
import pandas as pd
import random

API_ENDPOINT = "https://u7tivljb4txpkjnxhpzbydjlna0ynxka.lambda-url.eu-north-1.on.aws/predict"

st.set_page_config(page_title="RL Traffic Optimizer", layout="wide")

st.title("🚦 Autonomous Traffic Flow Optimizer")

st.info("The RL agent was trained using PPO (Proximal Policy Optimization) on a custom Gymnasium environment. At each step, it observes queue lengths across all 4 lanes and decides which signal phase minimizes total wait time — unlike fixed timers that cycle blindly regardless of traffic.")

phase_names = {
    0: "🟢 North-South (Straight)",
    1: "🟢 East-West (Straight)",
    2: "🟢 North-South (Left Turn)",
    3: "🟢 East-West (Left Turn)"
}

# Initialize session state
if "sim_step" not in st.session_state:
    st.session_state.sim_step = 0
    init_queues = [float(random.randint(2, 5)) for _ in range(4)]
    st.session_state.sim_queues = init_queues.copy()
    st.session_state.sim_fixed_queues = init_queues.copy()
    st.session_state.sim_active_phase = 0
    st.session_state.sim_total_routed = 0
    st.session_state.sim_time_on_phase = 0
    st.session_state.sim_rl_history = [sum(init_queues)]
    st.session_state.sim_fixed_history = [sum(init_queues)]
    st.session_state.sim_table_data = []
    st.session_state.sim_prev_total_queue = sum(init_queues)
    st.session_state.sim_prev_avg_queue = sum(init_queues)
    st.session_state.sim_queue_sum_accumulator = sum(init_queues)

if st.session_state.get("sim_show_reset_msg", False):
    st.success("Simulation reset! Click Step Forward to begin.")
    st.session_state.sim_show_reset_msg = False

# Sidebar
st.sidebar.header("Simulation Settings")

if st.sidebar.button("Reset Simulation"):
    for key in list(st.session_state.keys()):
        if key.startswith("sim_"):
            del st.session_state[key]
    st.session_state.sim_show_reset_msg = True
    st.rerun()

st.sidebar.markdown("### About")
st.sidebar.markdown("PPO-trained RL agent controlling a 4-way intersection")
st.sidebar.markdown("**Stack:** Python · Gymnasium · Stable-Baselines3 · FastAPI · AWS Lambda · Docker")

# Metric cards
col1, col2, col3 = st.columns(3)
current_total = sum(st.session_state.sim_queues)
avg_queue = st.session_state.sim_queue_sum_accumulator / max(1, st.session_state.sim_step + 1)

with col1:
    st.metric("Total Cars Routed", int(st.session_state.sim_total_routed))
with col2:
    st.metric(
        "Current Total Queue",
        int(current_total),
        delta=int(current_total - st.session_state.sim_prev_total_queue),
        delta_color="inverse"
    )
with col3:
    st.metric(
        "Avg Queue Per Step",
        f"{avg_queue:.1f}",
        delta=f"{avg_queue - st.session_state.sim_prev_avg_queue:.1f}",
        delta_color="inverse"
    )

st.markdown("---")

# Current step info
st.write(f"### Step: {st.session_state.sim_step} | Active Signal: {phase_names.get(st.session_state.sim_active_phase, 'Unknown')}")
st.caption(f"⏱ Time on current signal: {st.session_state.sim_time_on_phase * 5} seconds")

q_col1, q_col2, q_col3, q_col4 = st.columns(4)
q_col1.metric("North-South", int(st.session_state.sim_queues[0]))
q_col2.metric("East-West", int(st.session_state.sim_queues[1]))
q_col3.metric("N-S Left Turn", int(st.session_state.sim_queues[2]))
q_col4.metric("E-W Left Turn", int(st.session_state.sim_queues[3]))

# Step Forward button
if st.button("Step Forward (RL Agent)", type="primary"):

    # Generate arrivals using same Poisson-like logic as env
    # arrival_rates = [0.8, 0.8, 0.3, 0.3] → avg 0.8 and 0.3 cars per step
    arrivals = [
        int(random.random() < 0.8) + int(random.random() < 0.8),
        int(random.random() < 0.8) + int(random.random() < 0.8),
        int(random.random() < 0.3) + int(random.random() < 0.3),
        int(random.random() < 0.3) + int(random.random() < 0.3)
    ]

    # Apply arrivals first — send post-arrival state to API
    queues_after_arrivals = [q + a for q, a in zip(st.session_state.sim_queues, arrivals)]

    # Build payload matching new 9-float obs (no time_since_green)
    payload = {
        "queues": queues_after_arrivals,
        "current_phase": st.session_state.sim_active_phase,
        "time_on_phase": st.session_state.sim_time_on_phase
    }

    try:
        response = requests.post(API_ENDPOINT, json=payload)
        if response.status_code == 200:
            rl_phase = response.json().get("phase", 0)
        else:
            st.error(f"API Error: {response.text}")
            rl_phase = st.session_state.sim_active_phase
    except Exception as e:
        st.error(f"Failed to connect to backend: {e}")
        rl_phase = st.session_state.sim_active_phase

    # Update phase timer
    if rl_phase == st.session_state.sim_active_phase:
        st.session_state.sim_time_on_phase += 1
    else:
        st.session_state.sim_time_on_phase = 1

    st.session_state.sim_active_phase = rl_phase

    # Apply departures
    rl_departures = min(queues_after_arrivals[rl_phase], 3)
    queues_after_arrivals[rl_phase] -= rl_departures
    st.session_state.sim_queues = queues_after_arrivals
    st.session_state.sim_total_routed += rl_departures

    # Fixed timer — same arrivals for fair comparison
    fixed_phase = (st.session_state.sim_step // 3) % 4
    fixed_queues_after_arrivals = [
        q + a for q, a in zip(st.session_state.sim_fixed_queues, arrivals)
    ]
    fixed_departures = min(fixed_queues_after_arrivals[fixed_phase], 3)
    fixed_queues_after_arrivals[fixed_phase] -= fixed_departures
    st.session_state.sim_fixed_queues = fixed_queues_after_arrivals

    # State updates
    st.session_state.sim_step += 1
    new_total_queue = sum(st.session_state.sim_queues)

    st.session_state.sim_rl_history.append(new_total_queue)
    st.session_state.sim_fixed_history.append(sum(st.session_state.sim_fixed_queues))

    st.session_state.sim_prev_total_queue = current_total
    st.session_state.sim_prev_avg_queue = avg_queue
    st.session_state.sim_queue_sum_accumulator += new_total_queue

    step_record = {
        "Step": st.session_state.sim_step,
        "Active Signal": phase_names[rl_phase],
        "N-S Queue": int(st.session_state.sim_queues[0]),
        "E-W Queue": int(st.session_state.sim_queues[1]),
        "N-S Left Queue": int(st.session_state.sim_queues[2]),
        "E-W Left Queue": int(st.session_state.sim_queues[3]),
        "Total Queue": int(new_total_queue),
        "Cars Routed This Step": int(rl_departures)
    }
    st.session_state.sim_table_data.append(step_record)

    st.rerun()

st.markdown("---")

# Comparison chart
st.markdown("### RL Agent vs Fixed Timer — Total Queue Length Over Time")
chart_df = pd.DataFrame({
    "RL Agent": st.session_state.sim_rl_history,
    "Fixed Timer": st.session_state.sim_fixed_history
})
st.line_chart(chart_df)

# Step history table
with st.expander("Step-by-Step History"):
    if st.session_state.sim_table_data:
        history_df = pd.DataFrame(st.session_state.sim_table_data)
        st.dataframe(history_df.iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.write("No steps taken yet. Click 'Step Forward' to begin tracking.")