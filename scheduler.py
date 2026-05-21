import os
import time
import math
import random
import requests
import numpy as np
from datetime import datetime, timezone
from kubernetes import client, config, watch
from prometheus_client import start_http_server, Counter, Gauge
from stable_baselines3 import PPO

# 1. Prometheus Metrics Configuration
PROM_PORT = 9090  # Expose metrics on port 9090
PODS_SCHEDULED = Counter('carbon_scheduler_pods_scheduled_total', 'Total pods scheduled by the carbon-aware scheduler', ['node', 'sla'])
CARBON_EMISSIONS = Counter('carbon_scheduler_carbon_emissions_estimated_total', 'Estimated carbon emissions in grams of CO2')
SLA_VIOLATIONS = Counter('carbon_scheduler_sla_violations_total', 'Total SLA violations or forced schedules due to excessive delays')
NODE_UTILIZATION = Gauge('carbon_scheduler_node_utilization', 'Current node resource utilization ratio', ['node', 'resource'])
CARBON_INTENSITY = Gauge('carbon_scheduler_carbon_intensity', 'Current carbon intensity of node zone', ['node', 'zone'])

# 2. Scheduler Constants
SCHEDULER_NAME = "carbon-aware-scheduler"
NODE_ZONES = {
    "minikube": "us-east",
    "minikube-m02": "eu-west",
    "minikube-m03": "us-west"
}
P_IDLE = 100.0
P_MAX = 250.0
MAX_DELAY_STEPS = 5
CARBON_API_URL = os.getenv("CARBON_API_URL", "http://localhost:8000")

# Load cluster config
try:
    config.load_incluster_config()
    print("Running inside the Kubernetes cluster.")
except Exception:
    config.load_kube_config()
    print("Running outside the cluster (using local kubeconfig).")

v1 = client.CoreV1Api()

# 3. Helper Parsers for CPU and Memory
def parse_cpu(cpu_str):
    if not cpu_str:
        return 0.1
    if cpu_str.endswith('m'):
        return float(cpu_str[:-1]) / 1000.0
    return float(cpu_str)

def parse_memory(mem_str):
    if not mem_str:
        return 256.0
    if mem_str.endswith('Ki'):
        return float(mem_str[:-2]) / 1024.0
    if mem_str.endswith('Mi'):
        return float(mem_str[:-2])
    if mem_str.endswith('Gi'):
        return float(mem_str[:-2]) * 1024.0
    return float(mem_str) / (1024.0 * 1024.0)

# 4. Carbon Intensity Fetcher (with local simulation fallback)
def get_carbon_intensity(node_name, zone):
    try:
        r = requests.get(f"{CARBON_API_URL}/latest?zone={zone}", timeout=2.0)
        if r.status_code == 200:
            val = r.json()["carbonIntensity"]
            CARBON_INTENSITY.labels(node=node_name, zone=zone).set(val)
            return val
    except Exception as e:
        print(f"Error querying carbon API for {zone}: {e}. Falling back to simulation.")
    
    # Fallback simulation logic matching gym_env.py
    now = datetime.now(timezone.utc)
    hour = now.hour + now.minute / 60.0
    if zone == "us-east":
        val = 550.0 + random.uniform(-10.0, 10.0)
    elif zone == "eu-west":
        val = 250.0 + 80.0 * math.sin(hour / 4.0 * 2 * math.pi)
    elif zone == "us-west":
        val = 300.0 - 200.0 * max(0.0, math.sin(hour / 24.0 * 2 * math.pi))
    else:
        val = 300.0
        
    CARBON_INTENSITY.labels(node=node_name, zone=zone).set(val)
    return val

# 5. Bind Pod to Node API call
def bind_pod(name, namespace, node):
    body = client.V1Binding(
        metadata=client.V1ObjectMeta(name=name),
        target=client.V1ObjectReference(api_version="v1", kind="Node", name=node)
    )
    v1.create_namespaced_binding(namespace, body, _preload_content=False)

def get_node_resource_utilization():
    """
    Computes CPU and Memory utilization ratios of nodes based on
    pod requests currently scheduled on them.
    """
    nodes = v1.list_node().items
    sorted_nodes = sorted(nodes, key=lambda n: n.metadata.name)
    
    node_stats = {}
    for node in sorted_nodes:
        name = node.metadata.name
        cpu_capacity = parse_cpu(node.status.allocatable.get("cpu"))
        mem_capacity = parse_memory(node.status.allocatable.get("memory"))
        node_stats[name] = {
            "cpu_capacity": cpu_capacity,
            "mem_capacity": mem_capacity,
            "cpu_req_sum": 0.0,
            "mem_req_sum": 0.0
        }

    # Sum requests of all pods scheduled on nodes
    pods = v1.list_pod_for_all_namespaces().items
    for pod in pods:
        node_name = pod.spec.node_name
        if node_name and node_name in node_stats:
            for container in pod.spec.containers:
                reqs = container.resources.requests or {}
                node_stats[node_name]["cpu_req_sum"] += parse_cpu(reqs.get("cpu"))
                node_stats[node_name]["mem_req_sum"] += parse_memory(reqs.get("memory"))

    # Compute ratios
    ratios = {}
    for name, stats in node_stats.items():
        cpu_ratio = stats["cpu_req_sum"] / stats["cpu_capacity"]
        mem_ratio = stats["mem_req_sum"] / stats["mem_capacity"]
        ratios[name] = {
            "cpu_ratio": min(1.0, cpu_ratio),
            "mem_ratio": min(1.0, mem_ratio),
            "cpu_capacity": stats["cpu_capacity"]
        }
        NODE_UTILIZATION.labels(node=name, resource="cpu").set(min(1.0, cpu_ratio))
        NODE_UTILIZATION.labels(node=name, resource="memory").set(min(1.0, mem_ratio))

    return sorted_nodes, ratios

def get_pending_pods_stream():
    while True:
        try:
            pods = v1.list_pod_for_all_namespaces().items
            for pod in pods:
                yield pod
        except Exception as e:
            print(f"Error fetching pods: {e}")
        time.sleep(5)

# 6. Main Polling Loop
def main():
    # Start Prometheus server
    print(f"Starting Prometheus Metrics Exporter on port {PROM_PORT}...")
    start_http_server(PROM_PORT)

    # Load DRL Model
    model_path = "carbon_scheduler_model.zip"
    if not os.path.exists(model_path):
        print(f"Warning: {model_path} not found. Running training first...")
        # Fallback to train if model not found
        import train
        train.main()
    
    print("Loading DRL PPO model...")
    model = PPO.load(model_path)
    print("Model loaded successfully.")

    print(f"Polling for pending pods using schedulerName: {SCHEDULER_NAME}...")

    for pod in get_pending_pods_stream():
        # Filter for pending pods configured to use this scheduler and not yet bound
        if (pod.status.phase == 'Pending' and 
            pod.spec.scheduler_name == SCHEDULER_NAME and
            not pod.spec.node_name):
            
            name = pod.metadata.name
            namespace = pod.metadata.namespace
            annotations = pod.metadata.annotations or {}

            # Cooldown check for deferred pods
            delayed_until_str = annotations.get("carbon-scheduler.alpha.kubernetes.io/delayed-until")
            if delayed_until_str:
                try:
                    delayed_until = float(delayed_until_str)
                    if time.time() < delayed_until:
                        # Skip processing this pod for now (in cooldown)
                        continue
                except ValueError:
                    pass

            print(f"\nProcessing pod: {name} in namespace: {namespace}")
            
            # Fetch cluster state
            nodes, node_ratios = get_node_resource_utilization()
            node_names = [n.metadata.name for n in nodes]

            # Ensure we have exactly 3 nodes for model compatibility
            if len(node_names) < 3:
                print(f"Warning: Cluster has {len(node_names)} nodes. Duplicating nodes to match 3-node observation space.")
                # Pad node_names to size 3
                while len(node_names) < 3:
                    node_names.append(node_names[-1])

            # Gather carbon intensities
            carbon_vals = []
            for i, n_name in enumerate(node_names[:3]):
                n_obj = next((n for n in nodes if n.metadata.name == n_name), None)
                zone = n_obj.metadata.labels.get("zone", "us-east") if n_obj and n_obj.metadata.labels else "us-east"
                carbon_vals.append(get_carbon_intensity(n_name, zone))

            # Retrieve Pod resource requests
            pod_cpu_req = sum(parse_cpu(c.resources.requests.get("cpu") if c.resources.requests else None) for c in pod.spec.containers)
            pod_mem_req = sum(parse_memory(c.resources.requests.get("memory") if c.resources.requests else None) for c in pod.spec.containers)

            # Retrieve Pod SLA and current delay
            sla_str = annotations.get("carbon-scheduler.alpha.kubernetes.io/sla", "delay-tolerant")
            sla_class = 1 if sla_str == "latency-sensitive" else 0
            
            delay_str = annotations.get("carbon-scheduler.alpha.kubernetes.io/delay", "0")
            try:
                current_delay = int(delay_str)
            except ValueError:
                current_delay = 0

            # Formulate Gymnasium observation vector
            # Length 13: 3 * [cpu_ratio, mem_ratio, carbon_intensity_ratio] + [pod_cpu, pod_mem, sla, delay]
            obs = []
            for n_name in node_names[:3]:
                if n_name in node_ratios:
                    obs.extend([node_ratios[n_name]["cpu_ratio"], node_ratios[n_name]["mem_ratio"]])
                else:
                    obs.extend([0.0, 0.0])
                obs.append(min(1.0, carbon_vals[node_names.index(n_name)] / 800.0))

            max_cpu_capacity = max(node_ratios[n]["cpu_capacity"] for n in node_ratios) if node_ratios else 4.0
            obs.append(min(1.0, pod_cpu_req / max_cpu_capacity))
            obs.append(min(1.0, pod_mem_req / 8192.0))
            obs.append(float(sla_class))
            obs.append(min(1.0, current_delay / MAX_DELAY_STEPS))

            obs_array = np.array(obs, dtype=np.float32)

            # 7. Model Inference
            action, _ = model.predict(obs_array, deterministic=True)
            action = int(action)
            print(f"Observation vector: {obs_array}")
            print(f"Model recommended action: {action}")

            # 8. Action Execution and Safeguards
            # Action: Schedule on node
            if action < len(node_names):
                target_node = node_names[action]
                print(f"Action: Scheduling pod {name} on Node {target_node}...")
                
                # Bind the pod
                bind_pod(name, namespace, target_node)
                
                # Estimate emissions and update metrics
                target_node_obj = next((n for n in nodes if n.metadata.name == target_node), None)
                zone = target_node_obj.metadata.labels.get("zone", "us-east") if target_node_obj and target_node_obj.metadata.labels else "us-east"
                ci = carbon_vals[action]
                pod_power = (P_MAX - P_IDLE) * (pod_cpu_req / node_ratios[target_node]["cpu_capacity"])
                estimated_co2 = (pod_power / 1000.0) * 0.25 * ci # Grams of CO2
                
                PODS_SCHEDULED.labels(node=target_node, sla=sla_str).inc()
                CARBON_EMISSIONS.inc(estimated_co2)
                
                print(f"Pod {name} successfully bound to {target_node}. Estimated emissions: {estimated_co2:.4f}g CO2.")

            # Action: Defer
            else:
                # Safeguard 1: Latency-sensitive pods cannot be deferred!
                if sla_class == 1:
                    print("SLA Safeguard: Pod is latency-sensitive. Overriding deferral.")
                    # Force scheduling on least utilized CPU node
                    target_node = min(node_ratios, key=lambda k: node_ratios[k]["cpu_ratio"])
                    bind_pod(name, namespace, target_node)
                    PODS_SCHEDULED.labels(node=target_node, sla=sla_str).inc()
                    SLA_VIOLATIONS.inc()
                    print(f"Forced schedule pod {name} on node {target_node} due to SLA requirement.")
                
                # Safeguard 2: Delay-tolerant pods can only be deferred up to MAX_DELAY_STEPS
                else:
                    new_delay = current_delay + 1
                    if new_delay > MAX_DELAY_STEPS:
                        print("SLA Safeguard: Pod delay limit reached. Overriding deferral.")
                        target_node = min(node_ratios, key=lambda k: node_ratios[k]["cpu_ratio"])
                        bind_pod(name, namespace, target_node)
                        PODS_SCHEDULED.labels(node=target_node, sla=sla_str).inc()
                        SLA_VIOLATIONS.inc()
                        print(f"Forced schedule pod {name} on node {target_node} to prevent further delay SLA violations.")
                    else:
                        print(f"Action: Deferring scheduling for pod {name}. Delay count: {new_delay}/{MAX_DELAY_STEPS}")
                        # Update pod annotations
                        patch_body = {
                            "metadata": {
                                "annotations": {
                                    "carbon-scheduler.alpha.kubernetes.io/delay": str(new_delay),
                                    "carbon-scheduler.alpha.kubernetes.io/delayed-until": str(time.time() + 15.0) # Cooldown: 15s
                                }
                            }
                        }
                        v1.patch_namespaced_pod(name, namespace, body=patch_body)

if __name__ == "__main__":
    main()
