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
CLEAN_THRESHOLD = 250.0  # gCO2eq/kWh: below this, grid is clean enough to schedule immediately
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

def truncate_node_name(name):
    if not name:
        return name
    if "control-plane" in name:
        return "Control-Plane"
    if "stateful" in name:
        return "Worker-Stateful"
    if "stateless" in name:
        return "Worker-Stateless"
    return name

def record_decision(pod_name, sla, cpu, mem, action, target_node, reason):
    import json
    history_file = "/home/cc/Course Project/decision_history.json"
    data = []
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                data = json.load(f)
        except Exception:
            pass
    
    new_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pod_name": pod_name,
        "sla": sla,
        "cpu": cpu,
        "mem": mem,
        "action": action,
        "target_node": target_node,
        "reason": reason
    }
    data.append(new_entry)
    data = data[-100:]
    try:
        with open(history_file, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving decision: {e}")

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
                drl_target_node = node_names[action]
                
                # STRICT CARBON OVERRIDE
                # The user wants absolute priority on carbon, overriding the DRL's resource balancing.
                zone_intensities = {}
                zone_to_node = {}
                for idx, n_name in enumerate(node_names[:3]):
                    target_node_obj = next((n for n in nodes if n.metadata.name == n_name), None)
                    zn = target_node_obj.metadata.labels.get("zone", "unknown") if target_node_obj and target_node_obj.metadata.labels else "unknown"
                    zone_intensities[zn] = carbon_vals[idx]
                    zone_to_node[zn] = n_name
                
                cleanest_zone = min(zone_intensities, key=zone_intensities.get)
                cleanest_ci = zone_intensities[cleanest_zone]
                cleanest_node = zone_to_node[cleanest_zone]
                
                # If DRL chose a dirtier node AND the cleanest node has CPU < 85%
                if drl_target_node != cleanest_node and node_ratios[cleanest_node]["cpu_ratio"] < 0.85:
                    print(f"Strict Carbon Override: DRL chose {drl_target_node}, but {cleanest_node} is cleaner. Overriding.")
                    target_node = cleanest_node
                    action = node_names.index(cleanest_node) # Update action for accurate logging
                    override_used = True
                else:
                    target_node = drl_target_node
                    override_used = False

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
                
                # Record decision
                zone_intensities = {}
                for idx, n_name in enumerate(node_names[:3]):
                    target_node_obj = next((n for n in nodes if n.metadata.name == n_name), None)
                    zn = target_node_obj.metadata.labels.get("zone", "unknown") if target_node_obj and target_node_obj.metadata.labels else "unknown"
                    zone_intensities[zn] = carbon_vals[idx]
                
                selected_zone = zone
                selected_ci = ci
                min_zone = min(zone_intensities, key=zone_intensities.get)
                min_ci = zone_intensities[min_zone]
                
                # Build sorted zone ranking for clarity
                sorted_zones = sorted(zone_intensities.items(), key=lambda x: x[1])
                zone_ranking = ", ".join([f"{z}={c:.1f}" for z, c in sorted_zones])
                
                if override_used:
                    reason = (
                        f"Strict Carbon Override: DRL agent chose '{truncate_node_name(drl_target_node)}' for load balancing, "
                        f"but was forcefully overridden to '{truncate_node_name(target_node)}' ({selected_zone}) to strictly "
                        f"prioritize carbon reduction. "
                        f"Carbon ranking: [{zone_ranking}]. "
                        f"Node load: CPU {node_ratios[target_node]['cpu_ratio'] * 100:.1f}%, Memory {node_ratios[target_node]['mem_ratio'] * 100:.1f}%."
                    )
                elif selected_zone == min_zone or selected_ci <= min_ci + 15.0:
                    reason = (
                        f"Carbon-Optimal Placement: Scheduled on Node '{truncate_node_name(target_node)}' ({selected_zone}) "
                        f"because it has the lowest carbon intensity ({selected_ci:.1f} gCO2eq/kWh). "
                        f"Carbon ranking: [{zone_ranking}]. "
                        f"Node load: CPU {node_ratios[target_node]['cpu_ratio'] * 100:.1f}%, Memory {node_ratios[target_node]['mem_ratio'] * 100:.1f}%."
                    )
                else:
                    reason = (
                        f"Carbon-Aware Placement (with capacity check): Scheduled on Node '{truncate_node_name(target_node)}' ({selected_zone}, {selected_ci:.1f} gCO2eq/kWh). "
                        f"The cleanest zone ({min_zone}, {min_ci:.1f} gCO2eq/kWh) was at high CPU load, so the agent selected "
                        f"the next best option to balance carbon savings with resource availability. "
                        f"Carbon ranking: [{zone_ranking}]. "
                        f"Node load: CPU {node_ratios[target_node]['cpu_ratio'] * 100:.1f}%, Memory {node_ratios[target_node]['mem_ratio'] * 100:.1f}%."
                    )
                record_decision(name, sla_str, pod_cpu_req, pod_mem_req, "Scheduled", target_node, reason)

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
                    
                    zone_intensities = {}
                    for idx, n_name in enumerate(node_names[:3]):
                        target_node_obj = next((n for n in nodes if n.metadata.name == n_name), None)
                        zn = target_node_obj.metadata.labels.get("zone", "unknown") if target_node_obj and target_node_obj.metadata.labels else "unknown"
                        zone_intensities[zn] = carbon_vals[idx]
                    
                    reason = (
                        f"SLA Safeguard: Model recommended deferring (Action 3), but the pod is latency-sensitive (SLA: Fast) "
                        f"and cannot be delayed. Forced immediate scheduling on Node '{truncate_node_name(target_node)}' which has the lowest CPU utilization "
                        f"({node_ratios[target_node]['cpu_ratio'] * 100:.1f}%) to guarantee response time. "
                        f"Current grid carbon levels: " + ", ".join([f"{z}={c:.1f}" for z, c in zone_intensities.items()]) + "."
                    )
                    record_decision(name, sla_str, pod_cpu_req, pod_mem_req, "Forced Schedule", target_node, reason)
                
                # Safeguard 2: Delay-tolerant pods — smart deferral with clean threshold check
                else:
                    # First: collect zone intensities for decision making
                    zone_intensities = {}
                    zone_to_node = {}
                    for idx, n_name in enumerate(node_names[:3]):
                        target_node_obj = next((n for n in nodes if n.metadata.name == n_name), None)
                        zn = target_node_obj.metadata.labels.get("zone", "unknown") if target_node_obj and target_node_obj.metadata.labels else "unknown"
                        zone_intensities[zn] = carbon_vals[idx]
                        zone_to_node[zn] = n_name
                    
                    # Find the cleanest zone
                    cleanest_zone = min(zone_intensities, key=zone_intensities.get)
                    cleanest_ci = zone_intensities[cleanest_zone]
                    cleanest_node = zone_to_node[cleanest_zone]
                    sorted_zones = sorted(zone_intensities.items(), key=lambda x: x[1])
                    zone_ranking = ", ".join([f"{z}={c:.1f}" for z, c in sorted_zones])
                    
                    new_delay = current_delay + 1
                    
                    # SMART CHECK: If any zone is below the clean threshold, don't defer — schedule there!
                    if cleanest_ci < CLEAN_THRESHOLD:
                        print(f"Carbon Override: Zone {cleanest_zone} is clean ({cleanest_ci:.1f} < {CLEAN_THRESHOLD}). Scheduling immediately instead of deferring.")
                        target_node = cleanest_node
                        bind_pod(name, namespace, target_node)
                        PODS_SCHEDULED.labels(node=target_node, sla=sla_str).inc()
                        
                        target_node_obj = next((n for n in nodes if n.metadata.name == target_node), None)
                        zone = target_node_obj.metadata.labels.get("zone", "unknown") if target_node_obj and target_node_obj.metadata.labels else "unknown"
                        ci = zone_intensities[zone]
                        pod_power = (P_MAX - P_IDLE) * (pod_cpu_req / node_ratios[target_node]["cpu_capacity"])
                        estimated_co2 = (pod_power / 1000.0) * 0.25 * ci
                        CARBON_EMISSIONS.inc(estimated_co2)
                        
                        print(f"Pod {name} bound to {target_node}. Estimated emissions: {estimated_co2:.4f}g CO2.")
                        reason = (
                            f"Carbon-Optimal Override: DRL suggested deferral, but zone '{cleanest_zone}' is already clean "
                            f"({cleanest_ci:.1f} gCO2eq/kWh < threshold {CLEAN_THRESHOLD}). "
                            f"No benefit in waiting — scheduled immediately on Node '{truncate_node_name(target_node)}'. "
                            f"Carbon ranking: [{zone_ranking}]. "
                            f"Node load: CPU {node_ratios[target_node]['cpu_ratio'] * 100:.1f}%, Memory {node_ratios[target_node]['mem_ratio'] * 100:.1f}%."
                        )
                        record_decision(name, sla_str, pod_cpu_req, pod_mem_req, "Scheduled", target_node, reason)
                    
                    elif new_delay > MAX_DELAY_STEPS:
                        print("SLA Safeguard: Pod delay limit reached. Overriding deferral.")
                        target_node = cleanest_node  # Use cleanest node, not least-utilized
                        bind_pod(name, namespace, target_node)
                        PODS_SCHEDULED.labels(node=target_node, sla=sla_str).inc()
                        SLA_VIOLATIONS.inc()
                        print(f"Forced schedule pod {name} on node {target_node} to prevent further delay SLA violations.")
                        
                        reason = (
                            f"Temporal Shift Timeout: Pod waited {new_delay}/{MAX_DELAY_STEPS} steps but all zones remained "
                            f"above the clean threshold ({CLEAN_THRESHOLD} gCO2eq/kWh). "
                            f"Forced scheduling on the cleanest available node '{truncate_node_name(target_node)}' ({cleanest_zone}, {cleanest_ci:.1f} gCO2eq/kWh). "
                            f"Carbon ranking: [{zone_ranking}]."
                        )
                        record_decision(name, sla_str, pod_cpu_req, pod_mem_req, "Forced Schedule", target_node, reason)
                    else:
                        print(f"Action: Deferring scheduling for pod {name}. Delay count: {new_delay}/{MAX_DELAY_STEPS}. All zones above {CLEAN_THRESHOLD} gCO2eq/kWh.")
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
                        
                        reason = (
                            f"Temporal Shifting: Deferred scheduling (step {new_delay}/{MAX_DELAY_STEPS}) "
                            f"because ALL zones are above the clean threshold ({CLEAN_THRESHOLD} gCO2eq/kWh). "
                            f"Carbon ranking: [{zone_ranking}]. "
                            f"Waiting for grid conditions to improve before placing the workload."
                        )
                        record_decision(name, sla_str, pod_cpu_req, pod_mem_req, "Deferred", "-", reason)

if __name__ == "__main__":
    main()
