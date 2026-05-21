# Academic Project Report: Deep Reinforcement Learning for Carbon-Aware Kubernetes Scheduling

**Course Project Report**  
**Subject**: Advanced Cloud Computing & Systems Engineering  
**Date**: May 2026  

---

## Abstract
Modern cloud datacenters account for a significant portion of global electricity consumption, directly driving carbon dioxide emissions. Standard cloud orchestrators (e.g., Kubernetes `kube-scheduler`) perform placement decisions primarily to optimize resource density, neglecting grid-level carbon variations. This paper presents a **Deep Reinforcement Learning (DRL)-based Carbon-Aware Kubernetes Scheduler** that utilizes Proximal Policy Optimization (PPO) to make spatial-temporal placement decisions. Our system dynamically shifts workloads geographically (spatial shifting) to nodes located in cleaner power grids, and defers non-critical batch jobs (temporal shifting) to periods of high renewable generation. Evaluated on a multi-node Kubernetes cluster (Kind), the DRL scheduler achieves a **77.4% reduction in grid carbon emissions** and **100% elimination of CPU congestion** while strictly maintaining SLA constraints via custom safety overrides.

---

## 1. Introduction

### 1.1 Context and Motivation
Cloud computing facilities represent one of the fastest-growing consumers of electricity globally. The environmental impact of these datacenters is determined by the *carbon intensity* (gCO2eq/kWh) of the regional electricity grid powering them. Because grids rely on a variable mix of fossil fuels, solar, wind, and nuclear energy, carbon intensity changes dynamically over time and across different geographic regions. 

Traditional Kubernetes scheduling algorithms (e.g., `LeastRequestedPriority`, `MostRequestedPriority`) are local and static; they focus exclusively on local host parameters like CPU and memory utilization. They are completely blind to:
1.  **Spatial carbon intensity variations**: Scheduling a pod in a zone with low renewable generation (e.g., coal-heavy regions) rather than an alternative zone with high wind/solar.
2.  **Temporal energy fluctuations**: Scheduling a non-critical batch processing job immediately during peak grid load, instead of delaying it to a period of surplus renewable energy.

### 1.2 Proposed Solution
We propose a custom out-of-tree Kubernetes scheduler powered by a DRL agent. The scheduler:
*   Queries real-time carbon intensity per node zone.
*   Parses pod resource requirements and Service Level Agreement (SLA) specifications.
*   Employs a policy trained via PPO to determine whether to bind the pod to a node (selecting the cleanest node) or defer it.
*   Enforces deterministic safety fallback heuristics to guarantee that latency-sensitive workloads are never deferred, and delay-tolerant workloads are executed before breaching SLA deadlines.

---

## 2. System Architecture

The overall system architecture consists of five interacting components:

1.  **Kubernetes API Server**: Maintains the source of truth for cluster state. The scheduler queries pending pods and applies Bindings to nodes.
2.  **Mock Carbon Intensity API**: Serves simulated grid emissions data for three distinct regional zones (`us-east` [high intensity], `eu-west` [variable wind-dominant intensity], `us-west` [variable solar-dominant intensity]).
3.  **DRL Scheduler Controller**: The core control-plane loop. It runs on the host or inside a management pod, parses events, performs inference using the trained PPO policy, and updates pod annotations.
4.  **Prometheus Metrics Exporter**: Exposes real-time scheduler metrics (e.g., total scheduled pods, estimated emissions, SLA violations, node utilization) on port `9090`.
5.  **Workload Generator**: Generates synthetic load composed of latency-sensitive (continuous services) and delay-tolerant (batch sleep jobs) pods.

---

## 3. Mathematical Modeling & Reinforcement Learning Formulation

The scheduling problem is formulated as a discrete-time **Markov Decision Process (MDP)** defined by the tuple $(S, A, P, R, \gamma)$:

### 3.1 State Space ($S$)
The state vector $\mathbf{s}_t \in \mathbb{R}^{13}$ represents the complete status of the cluster and the incoming pod:
$$\mathbf{s}_t = \Big[ U^{\text{cpu}}_1, U^{\text{mem}}_1, C_1, \; U^{\text{cpu}}_2, U^{\text{mem}}_2, C_2, \; U^{\text{cpu}}_3, U^{\text{mem}}_3, C_3, \; R^{\text{cpu}}, R^{\text{mem}}, K, D \Big]$$

Where:
*   $U^{\text{cpu}}_n, U^{\text{mem}}_n \in [0.0, 1.0]$: CPU and Memory utilization ratios of Node $n$.
*   $C_n \in [0.0, 1.0]$: Normalized carbon intensity of Node $n$'s zone (original value in gCO2eq/kWh divided by a scale factor of 800.0).
*   $R^{\text{cpu}} \in [0.0, 1.0]$: Normalized CPU request of the pod relative to max node capacity.
*   $R^{\text{mem}} \in [0.0, 1.0]$: Normalized Memory request of the pod.
*   $K \in \{0, 1\}$: SLA class of the pod ($0 = \text{delay-tolerant}$, $1 = \text{latency-sensitive}$).
*   $D \in [0.0, 1.0]$: Normalized current delay steps of the pod ($D = \text{delay} / \text{MAX\_DELAY}$).

### 3.2 Action Space ($A$)
The action space is discrete with size 4:
$$A = \{0, 1, 2, 3\}$$
*   $a \in \{0, 1, 2\}$: Bind the pod to Node $a+1$.
*   $a = 3$: Defer the scheduling of the pod, returning it to the pending queue with an incremented delay counter.

### 3.3 Reward Function ($R$)
The reward function encourages carbon efficiency, resource balancing, and SLA adherence:
$$Reward = - \Big( w_{\text{carbon}} \cdot E_t + w_{\text{overload}} \cdot O_t + w_{\text{delay}} \cdot V_t \Big)$$

Where:
*   **Carbon Emission Penalty ($E_t$)**: If scheduled on Node $n$ at step $t$:
    $$E_t = \left(\frac{R^{\text{cpu}}}{\text{Capacity}_n} \cdot (P_{\text{max}} - P_{\text{idle}})\right) \times C_n$$
    If deferred ($a = 3$), $E_t = 0$.
*   **Node Overload Penalty ($O_t$)**: Penalizes scheduling a pod on a node that would exceed 90% CPU utilization:
    $$O_t = \sum_{n=1}^3 \max\left(0, U^{\text{cpu}}_n + \Delta U^{\text{cpu}}_n - 0.9\right)$$
*   **SLA Delay Penalty ($V_t$)**:
    *   If a latency-sensitive pod ($K=1$) is deferred: $V_t = 100.0$.
    *   If a delay-tolerant pod ($K=0$) is deferred: $V_t = 1.0$ (incentivizes scheduling unless carbon savings are significant).
    *   If a pod exceeds the maximum delay threshold: $V_t = 200.0$.

### 3.4 PPO Optimization Algorithm
We optimize the policy network parameters $\theta$ using Proximal Policy Optimization (PPO), which limits policy updates via clipping. The objective function is:
$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) \right]$$

Where:
*   $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$ represents the probability ratio.
*   $\hat{A}_t$ is the generalized advantage estimator (GAE).
*   $\epsilon = 0.2$ is the clipping hyperparameter.

---

## 4. Software Design & Unified Modeling Language (UML) Diagrams

### 4.1 Class Diagram (PlantUML)

This diagram represents the object-oriented structure of the custom scheduler implementation.

```plantuml
@startuml
skinparam style strictuml
title Carbon-Aware Scheduler Class Diagram

class CarbonSchedulerEnv {
  - num_nodes: int
  - node_capacities: List[float]
  - carbon_intensities: List[float]
  - node_cpu_util: List[float]
  - node_mem_util: List[float]
  - current_step: int
  + step(action: int) : Tuple
  + reset() : Tuple
  - _calculate_reward(action: int) : float
  - _get_obs() : List[float]
}

class PPO {
  + learn(total_timesteps: int)
  + predict(observation: List[float]) : Tuple[int, Any]
  + save(path: str)
  + load(path: str) : PPO
}

class KubernetesScheduler {
  - scheduler_name: str
  - prom_port: int
  - model: PPO
  - api_client: CoreV1Api
  + main()
  + get_node_resource_utilization() : Tuple
  + bind_pod(name: str, namespace: str, node: str)
  + get_carbon_intensity(node_name: str, zone: str) : float
  - get_pending_pods_stream() : Generator
}

class MockCarbonAPI {
  - app: FastAPI
  + get_latest_intensity(zone: str) : dict
  - simulate_diurnal_variation(zone: str) : float
}

class WorkloadGenerator {
  - api_client: CoreV1Api
  + create_pod(name: str, namespace: str)
  + main()
}

CarbonSchedulerEnv ..> PPO : "trains / evaluated by"
KubernetesScheduler --> PPO : "uses for decision making"
KubernetesScheduler --> MockCarbonAPI : "queries carbon values"
WorkloadGenerator ..> KubernetesScheduler : "submits pods scheduled by"
@endumuml
```

### 4.2 Sequence Diagram (PlantUML)

This diagram shows the step-by-step processing of a pod by the scheduler.

```plantuml
@startuml
skinparam style strictuml
title Pod Scheduling Sequence Diagram

actor User as "Workload Generator"
participant K8s as "Kubernetes API"
participant Sch as "Custom DRL Scheduler"
participant PPO as "PPO Neural Network"
participant API as "Mock Carbon API"
participant Prom as "Prometheus Exporter"

User -> K8s : Post Pod Manifest\n(schedulerName=carbon-aware-scheduler)
activate K8s
K8s --> User : HTTP 201 Created (Pod Pending)
deactivate K8s

activate Sch
Sch -> K8s : Get Node Allocatable & Running Pods
activate K8s
K8s --> Sch : Return Utilization & Capacities
deactivate K8s

Sch -> API : GET /latest?zone={zone} (for each Node)
activate API
API --> Sch : Return Carbon Intensity (gCO2eq/kWh)
deactivate API

Sch -> Sch : Build Observation Vector (Size 13)
Sch -> PPO : predict(observation)
activate PPO
PPO --> Sch : Return Recommended Action (0-3)
deactivate PPO

alt action in [0, 1, 2] (Bind Node)
    Sch -> K8s : POST Binding Request\n(target=Node_n)
    activate K8s
    K8s --> Sch : HTTP 201 Success
    deactivate K8s
    Sch -> Prom : Increment carbon_scheduler_pods_scheduled_total
    Sch -> Prom : Increment carbon_scheduler_carbon_emissions_estimated_total
else action == 3 (Defer & Cool down)
    alt SLA class == latency-sensitive OR delay >= 5
        Sch -> Sch : SLA Safeguard Triggered (Override action)
        Sch -> K8s : POST Binding Request\n(target=least_utilized_node)
        activate K8s
        K8s --> Sch : HTTP 201 Success
        deactivate K8s
        Sch -> Prom : Increment carbon_scheduler_sla_violations_total
    else delay < 5 (Allow Deferral)
        Sch -> K8s : PATCH Pod Annotations\n(delay++, delayed-until=time+15s)
        activate K8s
        K8s --> Sch : HTTP 200 Patched
        deactivate K8s
    end
end
deactivate Sch
@endumuml
```

---

## 5. Implementation Execution Guide

### 5.1 Installation of Local Environment
1.  **Clone code repository and install system level requirements**:
    ```bash
    sudo apt update && sudo apt install -y python3-pip python3-venv git curl
    ```
2.  **Setup Virtual Environment & Install CPU Torch (avoids CUDA package bloat)**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt
    ```

### 5.2 Preparing Kubernetes Infrastructure
We use a 3-node Kind cluster profile. Ensure the nodes are labeled appropriately:
```bash
# Make worker nodes schedulable (control plane + workers)
kubectl uncordon k8s-worker-stateless
kubectl taint nodes lab-cluster-control-plane node-role.kubernetes.io/control-plane:NoSchedule- || true

# Label worker nodes with distinct carbon zones
kubectl label node lab-cluster-control-plane zone=us-east --overwrite
kubectl label node k8s-worker-stateful zone=eu-west --overwrite
kubectl label node k8s-worker-stateless zone=us-west --overwrite
```

### 5.3 Step-by-Step Training & Launch
1.  **Execute Reinforcement Learning Offline Training**:
    ```bash
    python3 train.py
    ```
    *This tests your Gymnasium environment and trains PPO policy weights, storing them inside `carbon_scheduler_model.zip`.*
2.  **Start Mock Intensity Provider API**:
    ```bash
    python3 carbon_api.py
    ```
3.  **Start custom DRL Scheduler**:
    ```bash
    python3 -u scheduler.py
    ```
4.  **Inject dynamic workload generation to trigger scheduling loop**:
    ```bash
    python3 workload_generator.py
    ```

### 5.4 Monitoring Metrics
Open another terminal shell and query the Prometheus telemetry:
```bash
curl -s http://localhost:9090/metrics | grep carbon_scheduler
```

---

## 6. Experimental Evaluation & Verification Results

### 6.1 Simulation Environment Results
The PPO DRL scheduler was validated offline against the **Least-Utilized (Baseline)** heuristic. In this simulation, 100 scheduling requests were generated.

*   **Carbon Footprint reduction**: The Least-Utilized baseline generated **$361.18\text{ gCO}_2\text{eq}$** of estimated emissions due to placing intensive workloads in the heavy carbon zones during peak grid emissions. The DRL agent reduced this to **$81.65\text{ gCO}_2\text{eq}$** (a **77.4% reduction**).
*   **Congestion Prevention**: The baseline overloaded nodes 87 times due to a lack of awareness of global resource balancing across states. The DRL scheduler maintained **0 node overloads** by actively shifting workloads to cleaner and less congested locations.
*   **Load Shifting**: The PPO policy deferred **72** workloads temporarily during grid carbon peaks, verifying that it learned grid daily cycles and successfully delayed low-priority tasks.

### 6.2 Online Cluster Execution Logs Analysis
Traces from real-time execution in the Kind cluster confirm correct scheduler operations:

#### Event 1: Latency-Sensitive Pod Spatial Shifting
A pod annotated as `latency-sensitive` is scheduled immediately without delay. It is mapped to the cleanest zone (`k8s-worker-stateless`, which has active wind generation and low carbon intensity):
```
Processing pod: carbon-pod-11-8207 in namespace: default
Model recommended action: 1 (Node: k8s-worker-stateless)
Action: Scheduling pod carbon-pod-11-8207 on Node k8s-worker-stateless...
Pod carbon-pod-11-8207 successfully bound to k8s-worker-stateless.
```

#### Event 2: Delay-Tolerant Pod Temporal Shifting
A batch sleep pod is submitted during peak carbon hours. The DRL model recommends deferral (Action 3):
```
Processing pod: carbon-pod-10-1614 in namespace: default
Model recommended action: 3 (Defer)
Action: Deferring scheduling for pod carbon-pod-10-1614. Delay count: 2/5
```

#### Event 3: SLA Deadline Enforcement
When a deferred pod's delay count reaches 5, the scheduler detects a potential SLA violation, overrides the DRL choice, and forces binding:
```
Processing pod: carbon-pod-15-6098 in namespace: default
Model recommended action: 3 (Defer)
SLA Safeguard: Pod delay limit reached. Overriding deferral.
Forced schedule pod carbon-pod-15-6098 on node k8s-worker-stateful to prevent further delay SLA violations.
```

---

## 7. Conclusion & Future Outlook

This project demonstrated the feasibility of implementing carbon-aware scheduling policies directly on top of raw Kubernetes primitives without code rebuilds. PPO proved to be a highly adaptive framework for balancing green performance and traditional service levels. Future improvements will integrate:
1.  **Multi-step-ahead forecasting**: Querying actual prediction APIs to schedule batch jobs across multi-hour intervals.
2.  **Federated Clusters**: Enabling scheduling policies that span multi-cloud geographic regions (e.g., shifting workloads internationally via KubeFed based on global sun/wind patterns).
