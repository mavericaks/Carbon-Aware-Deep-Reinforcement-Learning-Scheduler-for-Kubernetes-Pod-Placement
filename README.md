# 🌍 Carbon-Aware Deep Reinforcement Learning Kubernetes Scheduler

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Scheduler-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Dashboard-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Stable Baselines3](https://img.shields.io/badge/DRL-PPO-FF4B4B?style=for-the-badge&logo=pytorch)

A custom, intelligent Kubernetes scheduler that leverages **Deep Reinforcement Learning (Proximal Policy Optimization)** to drastically reduce the carbon footprint of cloud workloads. It achieves this through autonomous **Spatial Shifting** (routing workloads to geo-regions with cleaner energy) and **Temporal Shifting** (deferring delay-tolerant batch jobs until grid carbon intensity drops).

Included is a stunning real-time glassmorphic dashboard to visualize the neural agent's decision-making rationale, cluster load, and live carbon metrics.

---

## 📖 Literature Survey & Novelty

### The Problem with Traditional Schedulers
The default `kube-scheduler` is designed purely for resource efficiency—its primary scoring functions revolve around `LeastAllocated` or `BalancedResourceAllocation`. It has **zero awareness** of the environmental impact of the physical data center powering the nodes.

Recent industry attempts to build "Carbon-Aware" systems usually rely on rigid, rule-based heuristics (e.g., *always pick the node with the lowest carbon intensity*). However, rigid heuristics fail in complex, high-load environments because:
1. They blindly pile workloads onto the "cleanest" node until it crashes (`OutOfcpu`).
2. They cannot learn the long-term cascading consequences of their placement decisions.
3. They struggle to balance Service Level Agreements (SLAs) with environmental goals.

### Our Novel Approach
This project establishes a novel approach by modeling workload scheduling as a **Markov Decision Process (MDP)** and training a **Proximal Policy Optimization (PPO)** agent to solve it. 

**Novel Contributions:**
1. **SLA-Aware Temporal Shifting:** The agent learns to read pod SLA annotations. If a workload is `delay-tolerant`, the agent may choose to *defer* scheduling (leaving the pod in a `Pending` state with an increasing delay counter) if all available grids are currently powered by fossil fuels, waiting for a cleaner renewable window. Latency-sensitive workloads bypass this to guarantee QoS.
2. **Dynamic Penalty Optimization:** The agent's reward function penalizes high carbon emissions (`w_carbon = 5.0`) while simultaneously penalizing node overloads (`w_overload = 0.5`). Through 50,000+ training steps, the agent naturally learns the optimal trade-off boundary between carbon reduction and cluster stability.
3. **Strict Carbon Override:** A deterministic fallback layer that intercepts the DRL agent if it prioritizes load-balancing too heavily, guaranteeing that workloads land on the absolute cleanest node provided it has <85% CPU utilization.

---

## 🏗️ System Architecture

```mermaid
graph TD
    subgraph "Kubernetes Cluster (Simulated / Minikube)"
        API[Kube API Server]
        W1[Node: us-east / Coal Heavy]
        W2[Node: eu-west / Wind Heavy]
        W3[Node: us-west / Solar Heavy]
    end

    subgraph "Carbon-Aware Scheduling Engine"
        SCHED[Custom Python Scheduler<br/>(scheduler.py)]
        PPO[DRL PPO Model<br/>(carbon_scheduler_model.zip)]
        C_API[Load-Aware Carbon API<br/>(carbon_api.py)]
    end

    subgraph "User & Observability"
        DASH[FastAPI Dashboard<br/>(dashboard_app.py)]
        PROM[Prometheus Exporter<br/>Port 9090]
        USER((User))
    end

    USER -->|Injects Workloads| DASH
    DASH -->|Creates Pods| API
    API -->|Watches Pending Pods| SCHED
    SCHED -->|Fetches Live Node Stats| API
    SCHED -->|Fetches Grid Intensity| C_API
    C_API -.->|Reads Node Load| API
    SCHED <-->|Passes State Vector| PPO
    SCHED -->|Executes Pod Binding| API
    SCHED -->|Logs Decision Rationale| DASH
    SCHED -->|Emits Metrics| PROM
```

---

## 📂 Project Structure

```text
.
├── carbon_api.py               # Simulated load-aware carbon grid intensity API
├── dashboard_app.py            # FastAPI backend for the interactive UI playground
├── templates/
│   └── index.html              # Stunning glassmorphic frontend for real-time visualization
├── gym_env.py                  # Custom Gymnasium environment defining the DRL MDP & rewards
├── train.py                    # Training script used to train the PPO agent
├── scheduler.py                # The core Kubernetes controller and DRL inference loop
├── carbon_scheduler_model.zip  # The pre-trained neural network (Ready to use)
├── prometheus.yml              # Prometheus configuration for metrics scraping
└── README.md                   # This documentation
```

---

## 🚀 Step-by-Step Execution Guide

To run this project locally, ensure you have a running Kubernetes cluster (e.g., `kind` or `minikube`) and your `~/.kube/config` is configured.

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Requires: `kubernetes`, `fastapi`, `uvicorn`, `stable-baselines3`, `prometheus-client`, `requests`)*

### 2. Start the Required Services
You must start three separate background services. Open a terminal and run the following commands sequentially:

```bash
# Start the Load-Aware Carbon API (Port 8000)
nohup python3 -u carbon_api.py > carbon_api.log 2>&1 &

# Start the Custom DRL Scheduler (Metrics on Port 9090)
nohup python3 -u scheduler.py > scheduler.log 2>&1 &

# Start the Interactive Web Dashboard (Port 8080)
nohup python3 dashboard_app.py > dashboard.log 2>&1 &
```

### 3. Access the Dashboard
Open your web browser and navigate to:
👉 **http://localhost:8080**

### 4. Inject Workloads and Observe
1. Use the **Manual Pod Injector** on the dashboard to create workloads.
2. Select **"Delay-Tolerant"** to watch the temporal shifting engine defer the pod if the grid is dirty.
3. Select **"Latency-Sensitive"** to watch it bypass deferrals and execute a Spatial Shift.
4. Watch the **Cluster Nodes State** graph dynamically react: as CPU load increases, the grid's carbon intensity will physically ramp up (simulating the firing up of peaker fossil-fuel plants).
5. Read the **Deep DRL Decisions & Rationale** to literally see the "thoughts" of the neural agent as it places your pods.

### 5. Monitor via Terminal (Optional)
If you prefer raw Kubernetes CLI output, you can watch the scheduler physically bind pods to nodes in real-time:
```bash
kubectl get pods -l app=carbon-workload -o wide -w
```
You can also view the raw, streaming rationale logs from the scheduler:
```bash
tail -f scheduler.log
```

---

## 🧹 Cleanup
To tear down the environment, you can click the **Reset Playground** button in the dashboard, or run the following kill commands:
```bash
pkill -f carbon_api.py
pkill -f scheduler.py
pkill -f dashboard_app.py
kubectl delete pods -l app=carbon-workload --force --grace-period=0
```
