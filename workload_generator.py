import time
import random
from kubernetes import client, config

def create_pod(v1, name, namespace="default"):
    # Randomize CPU and memory requests
    cpu_req = f"{random.randint(100, 500)}m"      # 0.1 to 0.5 CPU
    mem_req = f"{random.randint(128, 512)}Mi"     # 128MB to 512MB
    
    # SLA class: 60% delay-tolerant (batch), 40% latency-sensitive (real-time)
    sla_type = random.choice(["delay-tolerant", "latency-sensitive"])
    
    image = "nginx:alpine" if sla_type == "latency-sensitive" else "alpine:latest"
    command = None if sla_type == "latency-sensitive" else ["sleep", "60"]

    print(f"Creating Pod: {name} (CPU: {cpu_req}, Mem: {mem_req}, SLA: {sla_type})")

    pod_manifest = client.V1Pod(
        api_version="v1",
        kind="Pod",
        metadata=client.V1ObjectMeta(
            name=name,
            labels={"app": "carbon-workload", "sla": sla_type},
            annotations={
                "carbon-scheduler.alpha.kubernetes.io/sla": sla_type,
                "carbon-scheduler.alpha.kubernetes.io/delay": "0"
            }
        ),
        spec=client.V1PodSpec(
            scheduler_name="carbon-aware-scheduler",
            restart_policy="Never" if sla_type == "delay-tolerant" else "Always",
            containers=[
                client.V1Container(
                    name="workload-container",
                    image=image,
                    command=command,
                    resources=client.V1ResourceRequirements(
                        requests={"cpu": cpu_req, "memory": mem_req},
                        limits={"cpu": cpu_req, "memory": mem_req}
                    )
                )
            ]
        )
    )

    v1.create_namespaced_pod(namespace=namespace, body=pod_manifest)

def main():
    try:
        config.load_incluster_config()
    except Exception:
        config.load_kube_config()

    v1 = client.CoreV1Api()
    
    print("Starting Workload Generator...")
    pod_count = 0
    
    try:
        while pod_count < 20:
            pod_count += 1
            name = f"carbon-pod-{pod_count}-{random.randint(1000, 9999)}"
            create_pod(v1, name)
            # Sleep 5-15 seconds between submissions
            sleep_time = random.uniform(5, 15)
            print(f"Sleeping for {sleep_time:.1f} seconds...\n")
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("Workload generator stopped.")

if __name__ == "__main__":
    main()
