import time
import math
import random
import requests
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from kubernetes import client, config

app = FastAPI(title="Mock Carbon Intensity API")

# Load Kubernetes config to read real cluster load
try:
    config.load_incluster_config()
except Exception:
    config.load_kube_config()

v1 = client.CoreV1Api()

# Zone-to-node mapping (reverse lookup)
ZONE_NODE_MAP = {
    "us-east": "lab-cluster-control-plane",
    "eu-west": "k8s-worker-stateful",
    "us-west": "k8s-worker-stateless",
}

# Base carbon intensity per zone (represents the region's power grid mix)
# us-east: Coal-heavy grid → high base
# eu-west: Wind-heavy grid → low base
# us-west: Solar-heavy grid → medium base
ZONE_BASE_INTENSITY = {
    "us-east": 180.0,    # Cleanest baseline when idle
    "eu-west": 120.0,    # Cleanest grid (wind-powered)
    "us-west": 150.0,    # Medium baseline (solar)
}

# Maximum additional intensity from load (represents power plant ramp-up)
# When a node is at 100% CPU, the grid has to burn more fossil fuels to meet demand
ZONE_LOAD_MULTIPLIER = {
    "us-east": 450.0,    # Coal plants ramp aggressively
    "eu-west": 200.0,    # Wind regions ramp less
    "us-west": 300.0,    # Solar regions moderate ramp
}


def _parse_cpu(cpu_str):
    """Parse Kubernetes CPU string to float cores."""
    if not cpu_str:
        return 0.0
    if cpu_str.endswith('m'):
        return float(cpu_str[:-1]) / 1000.0
    return float(cpu_str)


def _parse_memory(mem_str):
    """Parse Kubernetes memory string to MiB."""
    if not mem_str:
        return 0.0
    if mem_str.endswith('Ki'):
        return float(mem_str[:-2]) / 1024.0
    if mem_str.endswith('Mi'):
        return float(mem_str[:-2])
    if mem_str.endswith('Gi'):
        return float(mem_str[:-2]) * 1024.0
    return float(mem_str) / (1024.0 * 1024.0)


def get_node_cpu_utilization(node_name: str) -> float:
    """
    Calculate the actual CPU request allocation ratio of a specific node
    by summing all pod CPU requests assigned to that node.
    Returns a ratio 0.0 - 1.0.
    """
    try:
        # Get node allocatable capacity
        node = v1.read_node(name=node_name)
        cpu_capacity = _parse_cpu(node.status.allocatable.get("cpu", "4"))

        # Sum CPU requests of all pods on this node
        all_pods = v1.list_pod_for_all_namespaces().items
        total_cpu_requests = 0.0
        for pod in all_pods:
            if pod.spec.node_name == node_name and pod.status.phase in ("Running", "Pending"):
                for container in pod.spec.containers:
                    if container.resources and container.resources.requests:
                        total_cpu_requests += _parse_cpu(container.resources.requests.get("cpu"))

        ratio = total_cpu_requests / cpu_capacity if cpu_capacity > 0 else 0.0
        return min(1.0, ratio)
    except Exception as e:
        print(f"Warning: Could not read node {node_name} utilization: {e}")
        return 0.0


def calculate_carbon_intensity(zone: str) -> float:
    """
    Carbon Intensity = Base Intensity + (Load Ratio * Load Multiplier) + small fluctuation

    Physics rationale:
    - When a data center is idle, the grid supplies baseline power (often renewables).
    - As CPU load increases, the grid must activate "peaker" plants (gas/coal) to meet
      the extra power demand. This directly increases the carbon intensity of the
      electricity being consumed.
    - A small random fluctuation (±10) simulates real-world grid variability.
    """
    if zone not in ZONE_BASE_INTENSITY:
        raise HTTPException(status_code=404, detail=f"Zone '{zone}' not found. Supported zones: us-east, eu-west, us-west")

    node_name = ZONE_NODE_MAP.get(zone)
    load_ratio = get_node_cpu_utilization(node_name) if node_name else 0.0

    base = ZONE_BASE_INTENSITY[zone]
    load_penalty = load_ratio * ZONE_LOAD_MULTIPLIER[zone]
    fluctuation = 0.0  # Removed random noise for clear correlation

    intensity = base + load_penalty + fluctuation
    return max(base * 0.8, intensity)  # Never go below 80% of base


class CarbonIntensityResponse(BaseModel):
    zone: str
    carbonIntensity: float
    unit: str = "gCO2eq/kWh"
    datetime: str
    updatedAt: str
    isEstimated: bool = False
    nodeLoadRatio: float = 0.0


@app.get("/latest", response_model=CarbonIntensityResponse)
def get_latest(zone: str = Query(..., description="Electricity zone, e.g. us-east, eu-west, us-west")):
    node_name = ZONE_NODE_MAP.get(zone)
    load_ratio = get_node_cpu_utilization(node_name) if node_name else 0.0
    intensity = calculate_carbon_intensity(zone)
    now_str = datetime.now(timezone.utc).isoformat()
    return CarbonIntensityResponse(
        zone=zone,
        carbonIntensity=round(intensity, 2),
        datetime=now_str,
        updatedAt=now_str,
        nodeLoadRatio=round(load_ratio, 4)
    )


@app.get("/forecast")
def get_forecast(zone: str = Query(..., description="Electricity zone, e.g. us-east, eu-west, us-west")):
    """Simple forecast: returns current intensity projected forward."""
    current = calculate_carbon_intensity(zone)
    forecasts = []
    now = datetime.now(timezone.utc)

    for h in range(24):
        # Forecast assumes load stays constant, small drift
        drift = random.uniform(-5.0, 5.0) * h * 0.1
        forecasts.append({
            "carbonIntensity": round(current + drift, 2),
            "datetime": (now.replace(hour=(now.hour + h) % 24)).isoformat()
        })

    return {
        "zone": zone,
        "forecast": forecasts
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
