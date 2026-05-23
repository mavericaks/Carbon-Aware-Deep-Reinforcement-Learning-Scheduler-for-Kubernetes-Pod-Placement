import os
import subprocess
import socket
import time
import requests
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from kubernetes import client, config

app = FastAPI(title="Carbon-Aware Scheduler Dashboard")

# Load Kubernetes configuration
try:
    config.load_kube_config()
except Exception:
    config.load_incluster_config()
v1 = client.CoreV1Api()

# Ports definition
API_PORT = 8000
SCHEDULER_PORT = 9090
DASHBOARD_PORT = 8080

class CustomPodPayload(BaseModel):
    cpu: float
    mem: float
    sla: str

def is_port_open(port: int) -> bool:
    """Check if a local port is open."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(('127.0.0.1', port)) == 0

def run_background_cmd(cmd: List[str], log_file: str):
    """Run a command in the background and redirect output."""
    log_path = os.path.join(os.getcwd(), log_file)
    with open(log_path, "w") as f:
        subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setpgrp)

@app.get("/api/status")
def get_status():
    """Return status of background services."""
    return {
        "carbon_api": is_port_open(API_PORT),
        "scheduler": is_port_open(SCHEDULER_PORT)
    }

@app.post("/api/start-api")
def start_api():
    """Start carbon API."""
    if is_port_open(API_PORT):
        return {"status": "already_running"}
    run_background_cmd(["python3", "-u", "carbon_api.py"], "carbon_api.log")
    time.sleep(1.5)
    return {"status": "started" if is_port_open(API_PORT) else "failed"}

@app.post("/api/start-scheduler")
def start_scheduler():
    """Start scheduler."""
    if is_port_open(SCHEDULER_PORT):
        return {"status": "already_running"}
    run_background_cmd(["python3", "-u", "scheduler.py"], "scheduler.log")
    time.sleep(2.0)
    return {"status": "started" if is_port_open(SCHEDULER_PORT) else "failed"}

@app.post("/api/inject-workloads")
def inject_workloads():
    """Run load generator."""
    run_background_cmd(["python3", "-u", "workload_generator.py"], "workload.log")
    return {"status": "injected"}

@app.post("/api/inject-custom-pod")
def inject_custom_pod(payload: CustomPodPayload):
    """Create a pod with manual specifications."""
    pod_name = f"custom-pod-{int(time.time())}"
    pod_manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": pod_name,
            "labels": {
                "app": "carbon-workload"
            },
            "annotations": {
                "carbon-scheduler.alpha.kubernetes.io/sla": payload.sla,
                "carbon-scheduler.alpha.kubernetes.io/delay": "0"
            }
        },
        "spec": {
            "schedulerName": "carbon-aware-scheduler",
            "containers": [
                {
                    "name": "app-container",
                    "image": "alpine:latest",
                    "command": ["sleep", "600"],
                    "resources": {
                        "requests": {
                            "cpu": f"{int(payload.cpu * 1000)}m",
                            "memory": f"{int(payload.mem)}Mi"
                        }
                    }
                }
            ]
        }
    }
    try:
        v1.create_namespaced_pod(namespace="default", body=pod_manifest)
        return {"status": "success", "pod_name": pod_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clean-up")
def clean_up():
    """Clean up workloads, history, and stop services."""
    subprocess.run("pkill -f carbon_api.py || true", shell=True)
    subprocess.run("pkill -f scheduler.py || true", shell=True)
    subprocess.run("kubectl delete pods -l app=carbon-workload --force --grace-period=0 --wait=false || true", shell=True)
    
    # Delete decision history
    history_file = os.path.join(os.getcwd(), "decision_history.json")
    if os.path.exists(history_file):
        try:
            os.remove(history_file)
        except Exception:
            pass
            
    time.sleep(1.0)
    return {"status": "cleaned_up"}

class PodDeleteRequest(BaseModel):
    pod_name: str

@app.post("/api/delete-pod")
def delete_pod(req: PodDeleteRequest):
    """Delete a specific pod to manually reduce cluster load."""
    try:
        subprocess.run(f"kubectl delete pod {req.pod_name} --force --grace-period=0 --wait=false", shell=True)
        return {"status": "deleted", "pod_name": req.pod_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/pods")
def get_pods():
    """Get active test workloads."""
    try:
        res = subprocess.run(
            ["kubectl", "get", "pods", "-l", "app=carbon-workload", "-o", "json"],
            capture_output=True, text=True, check=True
        )
        data = json.loads(res.stdout)
        
        pods = []
        for item in data.get("items", []):
            name = item["metadata"]["name"]
            status = item["status"]["phase"]
            
            # Catch OutOfcpu state
            container_statuses = item["status"].get("containerStatuses", [])
            if container_statuses:
                container_status = container_statuses[0]
                waiting_reason = container_status.get("state", {}).get("waiting", {}).get("reason")
                if waiting_reason == "OutOfcpu":
                    status = "OutOfcpu"
            
            node = item["spec"].get("nodeName", "-")
            annotations = item["metadata"].get("annotations", {})
            sla = annotations.get("carbon-scheduler.alpha.kubernetes.io/sla", "delay-tolerant")
            delay = annotations.get("carbon-scheduler.alpha.kubernetes.io/delay", "0")
            
            # Fetch CPU/Mem request
            cpu_req = "-"
            mem_req = "-"
            try:
                resources = item["spec"]["containers"][0].get("resources", {}).get("requests", {})
                cpu_req = resources.get("cpu", "-")
                mem_req = resources.get("memory", "-")
            except Exception:
                pass
            
            pods.append({
                "name": name,
                "status": status,
                "node": node,
                "sla": sla,
                "delay": int(delay),
                "cpu": cpu_req,
                "mem": mem_req
            })
        
        # Sort custom pods first, then standard workloads
        pods.sort(key=lambda x: x["name"])
        return pods
    except Exception:
        return []

@app.get("/api/nodes")
def get_nodes():
    """Retrieve node allocatable capacities, sum requests, and query zones carbon."""
    node_zones = {
        "lab-cluster-control-plane": "us-east",
        "k8s-worker-stateful": "eu-west",
        "k8s-worker-stateless": "us-west"
    }
    
    # Query Carbon API
    carbon_vals = {}
    for node, zone in node_zones.items():
        try:
            res = requests.get(f"http://127.0.0.1:{API_PORT}/latest?zone={zone}", timeout=0.5)
            carbon_vals[node] = res.json().get("carbon_intensity", 300.0)
        except Exception:
            carbon_vals[node] = 300.0

    # Retrieve Node Capacity
    node_capacities = {}
    try:
        nodes = v1.list_node().items
        for node in nodes:
            name = node.metadata.name
            cpu_cap = float(node.status.allocatable.get("cpu", "4"))
            mem_str = node.status.allocatable.get("memory", "8192Mi")
            if mem_str.endswith("Ki"):
                mem_cap = float(mem_str[:-2]) / 1024.0
            elif mem_str.endswith("Mi"):
                mem_cap = float(mem_str[:-2])
            elif mem_str.endswith("Gi"):
                mem_cap = float(mem_str[:-2]) * 1024.0
            else:
                mem_cap = float(mem_str) / 1024.0 / 1024.0
            node_capacities[name] = {"cpu": cpu_cap, "mem": mem_cap}
    except Exception:
        node_capacities = {
            "lab-cluster-control-plane": {"cpu": 4.0, "mem": 8192.0},
            "k8s-worker-stateful": {"cpu": 4.0, "mem": 8192.0},
            "k8s-worker-stateless": {"cpu": 4.0, "mem": 8192.0}
        }

    # Sum Aggregate Pod Allocations per node
    node_allocs = {n: {"cpu": 0.0, "mem": 0.0} for n in node_zones}
    try:
        pods = v1.list_pod_for_all_namespaces().items
        for pod in pods:
            node_name = pod.spec.node_name
            if node_name in node_zones and pod.status.phase in ["Running", "Pending"]:
                for container in pod.spec.containers:
                    res = container.resources
                    if res and res.requests:
                        # CPU
                        cpu_req = res.requests.get("cpu", "0")
                        if cpu_req.endswith("m"):
                            cpu_val = float(cpu_req[:-1]) / 1000.0
                        else:
                            cpu_val = float(cpu_req)
                        # Memory
                        mem_req = res.requests.get("memory", "0")
                        if mem_req.endswith("Ki"):
                            mem_val = float(mem_req[:-2]) / 1024.0
                        elif mem_req.endswith("Mi"):
                            mem_val = float(mem_req[:-2])
                        elif mem_req.endswith("Gi"):
                            mem_val = float(mem_req[:-2]) * 1024.0
                        else:
                            mem_val = float(mem_req) / 1024.0 / 1024.0
                        node_allocs[node_name]["cpu"] += cpu_val
                        node_allocs[node_name]["mem"] += mem_val
    except Exception:
        pass

    result = []
    for name in node_zones:
        cap = node_capacities.get(name, {"cpu": 4.0, "mem": 8192.0})
        alloc = node_allocs.get(name, {"cpu": 0.0, "mem": 0.0})
        result.append({
            "name": name,
            "zone": node_zones[name],
            "cpu_util": min(1.0, alloc["cpu"] / cap["cpu"]) if cap["cpu"] > 0 else 0.0,
            "mem_util": min(1.0, alloc["mem"] / cap["mem"]) if cap["mem"] > 0 else 0.0,
            "carbon": carbon_vals.get(name, 300.0)
        })
    return result

@app.get("/api/decisions")
def get_decisions():
    """Retrieve decision history from the shared json file."""
    history_file = os.path.join(os.getcwd(), "decision_history.json")
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                data = json.load(f)
                return data[-15:] # Return last 15 items
        except Exception:
            pass
    return []

@app.get("/api/logs")
def get_logs():
    """Return latest lines from scheduler log file."""
    log_path = os.path.join(os.getcwd(), "scheduler.log")
    if not os.path.exists(log_path):
        return []
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
        return [line.strip() for line in lines[-25:]]
    except Exception:
        return []

@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    template_path = os.path.join(os.getcwd(), "templates", "index.html")
    if os.path.exists(template_path):
        with open(template_path, "r") as f:
            return f.read()
    return "<h3>Error: templates/index.html not found!</h3>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=DASHBOARD_PORT)
