import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CarbonAwareSchedEnv(gym.Env):
    """
    Custom Gymnasium Environment for Carbon-Aware Kubernetes Pod Scheduling.
    Simulates a cluster with N nodes, each mapped to a different geographical zone
    with dynamic carbon intensity profiles.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, num_nodes=3, max_steps=100):
        super().__init__()
        self.num_nodes = num_nodes
        self.max_steps = max_steps
        self.current_step = 0

        # Node Capacities: CPU in cores, Memory in MB
        self.node_cpu_capacities = np.array([4.0, 4.0, 4.0], dtype=np.float32)
        self.node_mem_capacities = np.array([8192.0, 8192.0, 8192.0], dtype=np.float32)

        # Node Power consumption constants (in Watts)
        self.p_idle = 100.0
        self.p_max = 250.0

        # Zone names corresponding to nodes
        self.zones = ["us-east", "eu-west", "us-west"]

        # Max delay steps for delay-tolerant pods
        self.max_delay_steps = 5

        # Action Space:
        # Action 0..N-1: Schedule on Node i
        # Action N: Defer/Delay scheduling (only valid for delay-tolerant/batch pods)
        self.action_space = spaces.Discrete(self.num_nodes + 1)

        # Observation Space:
        # For each node: [CPU utilization ratio, Mem utilization ratio, Carbon intensity ratio]
        # For the incoming pod: [CPU request ratio, Mem request ratio, SLA class, Current delay ratio]
        # Total observation size = N * 3 + 4
        obs_size = self.num_nodes * 3 + 4
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_size,),
            dtype=np.float32
        )

        self.reset()

    def get_carbon_intensity(self, zone_idx, step):
        """
        Simulate dynamic grid carbon intensities (gCO2eq / kWh).
        - Zone 0 (us-east): Fossil-heavy. High average, low volatility.
        - Zone 1 (eu-west): Wind-heavy. Medium average, high stochastic volatility.
        - Zone 2 (us-west): Solar-heavy. Diurnal cycle (dips during midday, peaks at night).
        """
        # We assume 1 step = 15 minutes. 96 steps = 24 hours.
        time_of_day = (step % 96) / 96.0 * 2 * np.pi

        if zone_idx == 0:  # us-east
            # Stable high carbon intensity
            base = 550.0
            noise = np.random.normal(0, 15.0)
            return max(400.0, base + noise)
        elif zone_idx == 1:  # eu-west
            # Windy, unpredictable peaks and valleys
            base = 250.0
            wind_cycle = 80.0 * np.sin(step / 10.0)  # wind pattern
            noise = np.random.normal(0, 25.0)
            return max(80.0, base + wind_cycle + noise)
        elif zone_idx == 2:  # us-west
            # Solar curve: low during the day (sin > 0), high during night
            base = 300.0
            solar_dip = -200.0 * max(0.0, np.sin(time_of_day))  # solar production during day
            noise = np.random.normal(0, 10.0)
            return max(40.0, base + solar_dip + noise)
        else:
            return 300.0

    def generate_random_pod(self):
        """
        Generate a random pod request:
        - CPU: 0.1 to 2.0 cores
        - Memory: 256MB to 4096MB
        - SLA: 0 (Delay-tolerant/Batch) or 1 (Latency-sensitive)
        """
        cpu = np.random.uniform(0.1, 2.0)
        mem = np.random.uniform(256.0, 4096.0)
        sla = np.random.choice([0, 1], p=[0.6, 0.4])  # 60% batch, 40% real-time
        return {
            "cpu": cpu,
            "mem": mem,
            "sla": sla,
            "delay": 0
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Reset Node utilizations
        self.node_cpu_allocations = np.zeros(self.num_nodes, dtype=np.float32)
        self.node_mem_allocations = np.zeros(self.num_nodes, dtype=np.float32)

        # Get initial carbon intensities
        self.current_carbon = np.array([
            self.get_carbon_intensity(i, self.current_step)
            for i in range(self.num_nodes)
        ], dtype=np.float32)

        # Get the first pod
        self.current_pod = self.generate_random_pod()

        return self._get_obs(), {}

    def _get_obs(self):
        obs = []
        # Add Node states
        for i in range(self.num_nodes):
            cpu_ratio = self.node_cpu_allocations[i] / self.node_cpu_capacities[i]
            mem_ratio = self.node_mem_allocations[i] / self.node_mem_capacities[i]
            # Max intensity for normalization = 800 gCO2eq/kWh
            carbon_ratio = min(1.0, self.current_carbon[i] / 800.0)
            obs.extend([cpu_ratio, mem_ratio, carbon_ratio])

        # Add Pod state
        max_cpu_req = max(self.node_cpu_capacities)
        max_mem_req = max(self.node_mem_capacities)
        obs.append(min(1.0, self.current_pod["cpu"] / max_cpu_req))
        obs.append(min(1.0, self.current_pod["mem"] / max_mem_req))
        obs.append(float(self.current_pod["sla"]))
        obs.append(min(1.0, self.current_pod["delay"] / self.max_delay_steps))

        return np.array(obs, dtype=np.float32)

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        w_carbon = 5.0    # Highest priority: Minimize carbon footprint
        w_overload = 0.5  # Lower priority: Allow slight load imbalances to achieve cleaner nodes
        w_sla = 10.0      # Critical priority: Do not violate Latency-Sensitive SLAs
        w_delay = 0.1

        pod_scheduled = False
        pod_deferred = False

        # Action: 0..N-1 (Schedule on Node i)
        if action < self.num_nodes:
            node_idx = action
            
            # Check capacity constraints
            new_cpu = self.node_cpu_allocations[node_idx] + self.current_pod["cpu"]
            new_mem = self.node_mem_allocations[node_idx] + self.current_pod["mem"]

            # Overload penalty if capacity is exceeded
            overload = False
            if new_cpu > self.node_cpu_capacities[node_idx]:
                reward -= w_overload * (new_cpu - self.node_cpu_capacities[node_idx])
                overload = True
            if new_mem > self.node_mem_capacities[node_idx]:
                reward -= w_overload * ((new_mem - self.node_mem_capacities[node_idx]) / 1024.0)
                overload = True

            # Schedule the pod (even if overloading in simulation, to proceed)
            self.node_cpu_allocations[node_idx] = new_cpu
            self.node_mem_allocations[node_idx] = new_mem
            pod_scheduled = True

            # Calculate Carbon Footprint
            # Kepler proportional model: Pod active power = (P_max - P_idle) * (cpu_req / cpu_capacity)
            pod_power = (self.p_max - self.p_idle) * (self.current_pod["cpu"] / self.node_cpu_capacities[node_idx]) # Watts
            pod_energy_kwh = (pod_power / 1000.0) * 0.25 # Assuming pod runs for 0.25 hours (15 min timestep)
            carbon_emission = pod_energy_kwh * self.current_carbon[node_idx] # grams of CO2
            reward -= w_carbon * carbon_emission

            info["scheduled_node"] = node_idx
            info["carbon_emission"] = carbon_emission
            info["overloaded"] = overload

        # Action: Defer/Delay
        else:
            # Latency-sensitive (SLA=1) cannot be deferred
            if self.current_pod["sla"] == 1:
                reward -= w_sla * 10.0  # Large SLA penalty
                # Force placement on node with lowest CPU utilization
                node_idx = np.argmin(self.node_cpu_allocations / self.node_cpu_capacities)
                self.node_cpu_allocations[node_idx] += self.current_pod["cpu"]
                self.node_mem_allocations[node_idx] += self.current_pod["mem"]
                pod_scheduled = True
                info["forced_scheduling"] = True
                info["scheduled_node"] = node_idx
            else:
                # Delay-tolerant pod
                self.current_pod["delay"] += 1
                if self.current_pod["delay"] > self.max_delay_steps:
                    # Delay threshold reached, must schedule now!
                    reward -= w_sla * 5.0  # Delay SLA violation penalty
                    # Force scheduling
                    node_idx = np.argmin(self.node_cpu_allocations / self.node_cpu_capacities)
                    self.node_cpu_allocations[node_idx] += self.current_pod["cpu"]
                    self.node_mem_allocations[node_idx] += self.current_pod["mem"]
                    pod_scheduled = True
                    info["forced_scheduling"] = True
                    info["scheduled_node"] = node_idx
                else:
                    # Successfully deferred
                    reward -= w_delay * self.current_pod["delay"]
                    pod_deferred = True
                    info["deferred"] = True

        # Advance step
        self.current_step += 1
        
        # Simulated resource decay (representing workloads finishing)
        # Randomly decay CPU/Memory utilization by a small factor
        self.node_cpu_allocations = np.maximum(0.0, self.node_cpu_allocations - np.random.uniform(0.05, 0.3, self.num_nodes))
        self.node_mem_allocations = np.maximum(0.0, self.node_mem_allocations - np.random.uniform(50.0, 300.0, self.num_nodes))

        # Update carbon intensities for next step
        self.current_carbon = np.array([
            self.get_carbon_intensity(i, self.current_step)
            for i in range(self.num_nodes)
        ], dtype=np.float32)

        # Get next pod if current one is scheduled
        if pod_scheduled:
            self.current_pod = self.generate_random_pod()
        
        # Check termination
        if self.current_step >= self.max_steps:
            terminated = True

        return self._get_obs(), float(reward), terminated, truncated, info
