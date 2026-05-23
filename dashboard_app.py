import os
import subprocess
import socket
import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI(title="Carbon-Aware Scheduler Dashboard")

# Ports definition
API_PORT = 8000
SCHEDULER_PORT = 9090
DASHBOARD_PORT = 8080

def is_port_open(port: int) -> bool:
    """Check if a local port is open (i.e. service is running)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(('127.0.0.1', port)) == 0

def run_background_cmd(cmd: List[str], log_file: str):
    """Run a command in the background and redirect output to a log file."""
    log_path = os.path.join(os.getcwd(), log_file)
    with open(log_path, "w") as f:
        subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setpgrp)

@app.get("/api/status")
def get_status():
    """Return status of key background components."""
    return {
        "carbon_api": is_port_open(API_PORT),
        "scheduler": is_port_open(SCHEDULER_PORT)
    }

@app.post("/api/start-api")
def start_api():
    """Start mock carbon API if not running."""
    if is_port_open(API_PORT):
        return {"status": "already_running"}
    run_background_cmd(["python3", "-u", "carbon_api.py"], "carbon_api.log")
    time.sleep(1.5)
    return {"status": "started" if is_port_open(API_PORT) else "failed"}

@app.post("/api/start-scheduler")
def start_scheduler():
    """Start custom scheduler if not running."""
    if is_port_open(SCHEDULER_PORT):
        return {"status": "already_running"}
    run_background_cmd(["python3", "-u", "scheduler.py"], "scheduler.log")
    time.sleep(2.0)
    return {"status": "started" if is_port_open(SCHEDULER_PORT) else "failed"}

@app.post("/api/inject-workloads")
def inject_workloads():
    """Run workload generator to submit workloads."""
    run_background_cmd(["python3", "-u", "workload_generator.py"], "workload.log")
    return {"status": "injected"}

@app.post("/api/clean-up")
def clean_up():
    """Stop API, Scheduler, and delete k8s workloads."""
    # Kill python scripts
    subprocess.run("pkill -f carbon_api.py || true", shell=True)
    subprocess.run("pkill -f scheduler.py || true", shell=True)
    # Delete pods
    subprocess.run("kubectl delete pods -l app=carbon-workload --wait=false || true", shell=True)
    
    # Wait briefly
    time.sleep(1.0)
    return {"status": "cleaned_up"}

@app.get("/api/pods")
def get_pods():
    """Retrieve all test pods and parse details."""
    try:
        # Run kubectl get pods -o json
        res = subprocess.run(
            ["kubectl", "get", "pods", "-l", "app=carbon-workload", "-o", "json"],
            capture_output=True, text=True, check=True
        )
        import json
        data = json.loads(res.stdout)
        
        pods = []
        for item in data.get("items", []):
            name = item["metadata"]["name"]
            status = item["status"]["phase"]
            
            # Check containers for details if available
            container_status = item["status"].get("containerStatuses", [{}])[0]
            if container_status.get("state", {}).get("waiting", {}).get("reason") == "OutOfcpu":
                status = "OutOfcpu"
            
            node = item["spec"].get("nodeName", "-")
            
            annotations = item["metadata"].get("annotations", {})
            sla = annotations.get("carbon-scheduler.alpha.kubernetes.io/sla", "delay-tolerant")
            delay = annotations.get("carbon-scheduler.alpha.kubernetes.io/delay", "0")
            delayed_until = annotations.get("carbon-scheduler.alpha.kubernetes.io/delayed-until", "-")
            
            pods.append({
                "name": name,
                "status": status,
                "node": node,
                "sla": sla,
                "delay": int(delay),
                "delayed_until": delayed_until
            })
        
        # Sort by name
        pods.sort(key=lambda x: int(x["name"].split("-")[-2]) if len(x["name"].split("-")) >= 3 and x["name"].split("-")[-2].isdigit() else x["name"])
        return pods
    except Exception as e:
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

# Embedded HTML dashboard template
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
