# 🌍 Carbon-Aware Kubernetes Scheduler

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Scheduler-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Dashboard-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Stable Baselines3](https://img.shields.io/badge/DRL-PPO-FF4B4B?style=for-the-badge&logo=pytorch)

This project replaces the default Kubernetes scheduler with a custom **Deep Reinforcement Learning (DRL)** agent. Instead of just balancing CPU and RAM, this scheduler actively works to reduce the carbon emissions of your cloud cluster.

It features a beautiful web dashboard that lets you inject test pods and watch the AI make scheduling decisions in real-time.

---

## 📖 Why We Built This (Novelty & Context)

### The Problem
The default Kubernetes scheduler only cares about hardware efficiency. It puts your applications on whichever node has the most available CPU or memory. It has no idea if that node is being powered by a dirty coal plant or a clean solar farm.

### Our Solution
We trained an AI model using Proximal Policy Optimization (PPO) to make scheduling decisions based on live energy grid carbon intensity. Our system introduces two major novelties:

1. **Spatial Shifting (Location):** If an application must run immediately, the scheduler places it on the node located in the region with the cleanest energy.
2. **Temporal Shifting (Time):** If an application is marked as "Delay-Tolerant" (like a background data job), the scheduler can choose to pause it if all grids are currently dirty, waiting for the wind to pick up or the sun to come out.

---

## 🏗️ System Flow Architecture

This diagram shows the sequential step-by-step flow of how a single pod is handled by our system:

```mermaid
graph LR
    %% Core Entities
    User(("User"))
    Dash["Web Dashboard"]
    KubeAPI["Kubernetes API"]
    CarbonAPI["Grid Carbon API"]
    
    %% Scheduler components
    subgraph "Scheduling Brain"
        Sched["Python Scheduler"]
        DRL["Trained PPO AI"]
    end
    
    %% Nodes
    subgraph "Cluster Nodes"
        N1["US-East<br/>(Coal Grid)"]
        N2["EU-West<br/>(Wind Grid)"]
    end
    
    %% Flow
    User -- "1. Clicks Deploy" --> Dash
    Dash -- "2. Creates Pod" --> KubeAPI
    KubeAPI -- "3. Sees Pending Pod" --> Sched
    Sched -- "4. Gets Carbon Data" --> CarbonAPI
    Sched -- "5. Asks for Prediction" --> DRL
    Sched -- "6. Binds Pod to Clean Node" --> KubeAPI
    KubeAPI -- "7. Starts Container" --> N2
```

---

## 📂 What's In This Repository

* `scheduler.py`: The core custom Kubernetes scheduler script.
* `carbon_api.py`: A simulated API that tracks live node CPU load and generates carbon intensity metrics.
* `dashboard_app.py` & `templates/`: The code for the interactive monitoring dashboard.
* `train.py` & `gym_env.py`: The machine learning scripts used to train the AI.
* `carbon_scheduler_model.zip`: The pre-trained AI brain (ready to go!).

---

## 🚀 How to Run It

You need a running Kubernetes cluster (like minikube) and `kubectl` configured.

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the 3 Core Services
Open your terminal and run these commands to start the background systems:

```bash
# 1. Start the Carbon Data API
nohup python3 -u carbon_api.py > carbon_api.log 2>&1 &

# 2. Start the AI Scheduler
nohup python3 -u scheduler.py > scheduler.log 2>&1 &

# 3. Start the Web Dashboard
nohup python3 dashboard_app.py > dashboard.log 2>&1 &
```

### 3. Open the Dashboard
Go to your web browser and open:
👉 **http://localhost:8080**

From here, you can click **"Inject Load Test Pods"** and watch the AI route them to the greenest available nodes!

### 4. How to Stop the System
When you are done playing with it, run these commands to clean up:
```bash
pkill -f carbon_api.py
pkill -f scheduler.py
pkill -f dashboard_app.py
kubectl delete pods -l app=carbon-workload --force --grace-period=0
```

---

## 🛠️ Troubleshooting: After VM Restart

If you suspend or restart your virtual machine, the simulated Kubernetes cluster nodes will lose their virtual network bridges and your pods will be stuck in a `Pending` state, and the web dashboard processes will die.

Run this recovery sequence:

**1. Fix the Kubernetes Cluster (Restart Docker):**
```bash
sudo systemctl restart docker
sleep 15
kubectl get nodes   # Ensure they say "Ready"
```

**2. Ensure the Dashboard Port is Free:**
*(If a rogue service like Nginx starts on port 8080 on boot, kill it)*
```bash
sudo fuser -k 8080/tcp
```

**3. Restart the Core Services:**
```bash
cd ~/Course\ Project
nohup python3 -u carbon_api.py > carbon_api.log 2>&1 &
nohup python3 -u scheduler.py > scheduler.log 2>&1 &
nohup python3 dashboard_app.py > dashboard.log 2>&1 &
```
