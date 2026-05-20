# 🚦 Autonomous Traffic Flow Optimizer

> PPO-trained RL agent that controls traffic signals at a 4-way intersection — dynamically outperforming fixed-timer systems by responding to real-time queue lengths.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://traffic-flow-optimizer-2fwlhjq7tvrrvpnd2woh5z.streamlit.app/)
[![AWS Lambda](https://img.shields.io/badge/Backend-AWS%20Lambda-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)](https://aws.amazon.com/lambda/)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/Kavyansh180/traffic-flow-optimizer/actions)

---

![App Demo](docs/demo.png)

---

## Results

![Chart](docs/chart.png)

| Metric | RL Agent | Fixed Timer |
|--------|----------|-------------|
| Avg Queue Per Step | 50.9 | 59.4 |
| Efficiency | **14.3% better** | baseline |
| Phase Distribution | Balanced (all 4 phases) | Fixed cycle |
| Total Cars Routed | 180 | — |

---

## What it does

Real-world traffic signals run on fixed timers regardless of actual traffic. This project trains a **PPO Reinforcement Learning agent** on a custom Gymnasium simulation to make smarter decisions — observing queue lengths across all 4 lanes every step and selecting which signal phase minimizes total wait time.

---

## Architecture

```
Streamlit Frontend  →  API Gateway  →  AWS Lambda (FastAPI + Docker)
                                              ↓
                                    PPO Model (baked in ECR image)
                                              ↓
                                    AWS CloudWatch (monitoring)

GitHub Actions → keep_alive.yml (pings app every 8hrs)
```

---

## Tech Stack

| Layer | Tech |
|-------|------|
| RL Algorithm | PPO — Stable-Baselines3 |
| Environment | Custom Gymnasium |
| Inference | FastAPI + Mangum |
| Deploy | AWS Lambda + ECR + Docker |
| Monitoring | AWS CloudWatch |
| Frontend | Streamlit Cloud |
| CI/CD | GitHub Actions |

---

## Project Structure

```
├── env/               # Custom Gymnasium environment
├── training/          # PPO training + MLflow tracking
├── inference/         # FastAPI app + Dockerfile
├── frontend/          # Streamlit dashboard
├── baseline/          # Fixed-timer baseline
└── .github/workflows/ # keep_alive + deploy pipelines
```

---

## Key Design Decisions

**PPO over DQN** — More stable with noisy reward signals, converged faster, better phase balance.

**Lambda over EC2** — Inference is stateless and bursty. Lambda costs zero when idle vs EC2 running 24/7.

**Model in Docker image** — Simplifies architecture and reduces cold start time for demo scale. Production would separate via S3.

**Poisson arrivals** — Real traffic follows Poisson distribution. Tuned λ=0.8 (main lanes), λ=0.3 (turn lanes) to make the environment actually solvable.

---

## Validation Results

```
Phase 0 (N-S Straight):   36.0% ✅
Phase 1 (E-W Straight):   35.5% ✅
Phase 2 (N-S Left Turn):  13.8% ✅
Phase 3 (E-W Left Turn):  14.7% ✅
```
All 4 phases selected — no lane starvation.

---

## GitHub Actions

![GitHub Actions](docs/github_actions.png)

`keep_alive.yml` pings the Streamlit app every 8 hours to prevent sleep.

---

## Quick Start

```bash
git clone https://github.com/Kavyansh180/traffic-flow-optimizer.git
cd traffic-flow-optimizer
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

# Train
python -m training.train

# Run frontend
cd frontend && streamlit run streamlit_app.py
```

---

**Author:** [Kavyansh Vats](https://github.com/Kavyansh180)
