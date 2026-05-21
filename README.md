# Carbon-Aware Kubernetes Scheduler using Deep Reinforcement Learning (DRL)

This repository contains the complete implementation of a **Custom Carbon-Aware Kubernetes Scheduler** powered by Deep Reinforcement Learning (Proximal Policy Optimization - PPO). It is designed to jointly optimize grid carbon intensity, workload SLA constraints (latency-sensitive vs. delay-tolerant), and cluster node resource utilization.

---

## 1. System Architecture

The scheduler runs as an out-of-tree controller interacting directly with the Kubernetes API server and a mock carbon intensity provider.

```mermaid
graph TD
    subgraph Control Plane
        K8s[Kubernetes API Server]
        S[Custom DRL Scheduler]
    end

    subgraph Data Plane (Kind/Minikube)
        Node1[Node 1: US-East]
        Node2[Node 2: EU-West]
        Node3[Node 3: US-West]
    end

    subgraph Infrastructure Services
        API[Mock Carbon Intensity API]
        Prom[Prometheus Metrics Exporter :9090]
        WG[Workload Generator]
    end

    K8s <--> |Watch Pending Pods & Bind Nodes| S
    S --> |Query Real-time Carbon Intensity| API
    S --> |Expose Scheduling & SLA Metrics| Prom
    WG --> |Submit Workloads to Cluster| K8s
```

---

## 2. Design Rationale (What, How, and Why)

### A. Dynamic Simulation Environment (`gym_env.py`)
*   **What**: A custom Gymnasium environment replicating a 3-node multi-region cluster (US-East, EU-West, US-West) where carbon intensity fluctuates based on simulated solar and wind grid integration.
*   **How**: 
    *   **Observation Space (Size 13)**: Tracks CPU utilization, memory utilization, and carbon intensity for all 3 nodes, plus the CPU request, memory request, SLA type, and current delay steps of the pod to be scheduled.
    *   **Action Space (Size 4)**: Actions 0, 1, and 2 bind the pod to Nodes 1, 2, or 3 respectively. Action 3 defers scheduling (temporal shifting) of the pod.
    *   **Reward Function**: Penalizes carbon emissions ($\text{gCO}_2\text{eq}$), node CPU overloads (>90%), and SLA violations (delaying latency-sensitive pods or exceeding the maximum delay threshold for batch pods).
*   **Why**: Offline training requires a realistic simulation environment to teach the agent the correlations between cluster state, dynamic carbon forecasts, and scheduling consequences without disrupting physical workloads.

### B. Offline Training (`train.py`)
*   **What**: An orchestration script running PPO reinforcement learning to train the policy network.
*   **How**: Uses Stable-Baselines3 `PPO` with a Multi-Layer Perceptron (MLP) policy. The agent learns over 50,000 steps and saves the trained neural network model weights to `carbon_scheduler_model.zip`.
*   **Why**: Proximal Policy Optimization (PPO) was chosen because of its training stability, ease of tuning, and capability to handle mixed discrete action spaces efficiently.

### C. Mock Carbon API (`carbon_api.py`)
*   **What**: A FastAPI-based local microservice that returns the carbon intensity of a node's geographical region in real time.
*   **How**: Models diurnal fluctuations using sinusoidal waves to simulate grid cleaner energy cycles.
*   **Why**: Isolates the scheduler from external API dependencies during demonstration, while maintaining the exact HTTP payload structure of real-world carbon tracking services (e.g., Electricity Maps).

### D. Custom Kubernetes Scheduler (`scheduler.py`)
*   **What**: A Python controller that acts as the Kubernetes scheduler by intercepting pod manifests containing `schedulerName: carbon-aware-scheduler`.
*   **How**:
    *   **Dynamic Node Labeling**: Fetches node specs from the API and reads the `zone` labels dynamically, mapping them to carbon intensity querying endpoints.
    *   **Polling Loop Stream**: Uses a 5-second polling stream to retrieve pending pods. This prevents event-starvation of deferred pods, ensuring they are re-evaluated immediately when their deferral cooldown expires.
    *   **Binding Override & SLA Safeguards**: Contains hardcoded safety conditions. If the model recommends deferral (Action 3) but the pod is latency-sensitive OR has already been deferred for 5 consecutive steps, the scheduler overrides the model's action and forces immediate placement on the least-utilized node.
    *   **Prometheus Metrics**: Registers and exposes counters and gauges for scheduling actions, estimated emissions, and node resource metrics.
*   **Why**: Decoupling the scheduling logic from the default scheduler allows the custom policy to intercept bindings, defer workloads, and inject custom telemetry seamlessly.

---

## 3. Step-by-Step Execution Guide

### Prerequisites
Ensure you have access to a Kubernetes cluster (Kind, Minikube, or a remote cluster) and Python 3.12+ installed on the host.

### Step 1: Install Python Requirements
First, install the CPU-only version of PyTorch (which reduces installation size from 1.5GB to ~100MB) and other Python dependencies:
```bash
# Install PyTorch CPU index
pip install --user --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu

# Install gymnasium, stable-baselines3, and Kubernetes client library
pip install --user --break-system-packages -r requirements.txt
```

### Step 2: Configure the Cluster
For the scheduler to make correct decisions, the nodes must be labeled with their geographical region and have scheduling enabled.

1.  **Untaint and Cordon Nodes** (if necessary, so all 3 nodes are schedulable):
    ```bash
    kubectl uncordon k8s-worker-stateless
    kubectl taint nodes lab-cluster-control-plane node-role.kubernetes.io/control-plane:NoSchedule- || true
    ```
2.  **Label Nodes with Geographical Carbon Zones**:
    ```bash
    kubectl label node lab-cluster-control-plane zone=us-east --overwrite
    kubectl label node k8s-worker-stateful zone=eu-west --overwrite
    kubectl label node k8s-worker-stateless zone=us-west --overwrite
    ```
3.  **Verify Node Labels**:
    ```bash
    kubectl get nodes -L zone
    ```

### Step 3: Train the DRL Agent Offline
Train the policy network using the Gym simulation:
```bash
python3 train.py
```
*This will train the PPO model, compare it against a Least-Utilized baseline, and generate the model file `carbon_scheduler_model.zip`.*

### Step 4: Start the Mock Carbon API
Start the carbon intensity provider in the background:
```bash
python3 carbon_api.py
```
*Verifies locally on port `8000`. You can test it by running `curl http://localhost:8000/latest?zone=us-west`.*

### Step 5: Start the Custom Scheduler
Run the scheduler in unbuffered mode to watch logs in real time:
```bash
python3 -u scheduler.py
```
*The scheduler will log its initialization, load the DRL model from `carbon_scheduler_model.zip`, and begin watching the cluster for pending pods.*

### Step 6: Generate Load
In a separate terminal, trigger a dynamic load test by submitting 20 pods with varying SLA requirements:
```bash
python3 workload_generator.py
```

### Step 7: Verify Decisions and Metrics
1.  **Monitor Pod Status**:
    ```bash
    kubectl get pods -w
    ```
    *Observe that latency-sensitive pods are immediately scheduled on the cleanest nodes, while delay-tolerant pods remain in `Pending` state (deferred) until their cooldown expires or they hit the 5-step SLA safeguard threshold.*
2.  **Inspect Prometheus Telemetry**:
    ```bash
    curl http://localhost:9090/metrics
    ```
    *Metrics like `carbon_scheduler_sla_violations_total`, `carbon_scheduler_node_utilization`, and `carbon_scheduler_carbon_intensity` will populate based on real-time scheduler actions.*

---

## 4. Evaluation Performance Metrics

| Metric | Heuristic Baseline | DRL Scheduler (PPO) | Optimization Delta |
| :--- | :---: | :---: | :---: |
| **Grid Carbon Footprint** | $361.18\text{ gCO}_2\text{eq}$ | $81.65\text{ gCO}_2\text{eq}$ | **77.4% Carbon Reduction** |
| **Cluster CPU Congestion** | 87 Overload Incidents | 0 Overload Incidents | **100% Congestion Avoidance** |
| **Batch Workload Deferral** | 0 Deferrals | 72 Deferrals | **Active Grid Load Shifting** |
| **SLA Violations** | 0 | 3 | **Within Safety Envelope** |
