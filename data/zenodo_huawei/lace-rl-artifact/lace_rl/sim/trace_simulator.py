from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class FunctionInvocation:
    timestamp: int
    pod_id: str
    region_id: str
    exec_time_s: float
    cpu_cores: float
    mem_MB: float
    cold_start_latency_s: float = 0.0
    user_region: str = 'local'

    metadata: dict = None

@dataclass
class GreenEnergy:
    timestamp: int
    region_id: str
    # green_ratio: float
    carbon_intensity: float

class EnergyEstimator:
    def __init__(self):
        # Simulate m5-series EC2 instance
        self.J_DRAM_per_MB = 0.00037
        self.J_CPU_per_mCore = 0.3
        # self.J_CPU_per_mCore = 0.188

        self.LAMBDA_IDLE = 0.2
        self.N_CPU = 32
        self.TOTAL_DRAM_MB = 16384

    def estimate_keep_alive_energy(self, mem_MB, cpu_cores, duration_s, carbon_intensity):
        cpu_frac = cpu_cores / self.N_CPU
        power = self.J_DRAM_per_MB * mem_MB + self.J_CPU_per_mCore * self.LAMBDA_IDLE * cpu_frac
        energy = power * duration_s
        carbon = self.estimate_carbon_emission(energy, carbon_intensity)
        return energy, carbon

    def estimate_exec_energy(self, exec_time_s, cpu_cores, mem_MB):
        cpu_frac = cpu_cores / self.N_CPU
        power = self.J_DRAM_per_MB * mem_MB + self.J_CPU_per_mCore * cpu_frac
        return power * exec_time_s

    def estimate_carbon_emission(self, energy_joule, carbon_intensity):
        energy_kwh = energy_joule / (3.6 * 1e6)
        return energy_kwh * carbon_intensity

class TraceDrivenSimulator:
    def __init__(self, green_trace: Dict[str, Dict[int, GreenEnergy]], network_latency_map=None, fixed_keep_alive_s=60.0):
        self.green_trace = green_trace
        self.fixed_keep_alive_s = fixed_keep_alive_s
        self.network_latency_map = network_latency_map or {}
        self.energy_estimator = EnergyEstimator()
        self.reset_metrics()

    def reset_metrics(self):
        self.container_state: Dict[str, Tuple[int, float]] = {}
        self.total_energy = 0.0
        self.total_carbon = 0.0
        self.keep_alive_energy = 0.0
        self.execution_energy = 0.0
        self.keep_alive_carbon = 0.0
        self.execution_carbon = 0.0
        self.cold_start_count = 0
        self.total_latency = 0.0
        self.total_network_latency = 0.0
        self.total_invocations = 0
        self.total_idle_time: Dict[str, float] = {}
        self.total_idle_energy: Dict[str, float] = {}

    def run(self, invocations: List[FunctionInvocation]):
        for inv in invocations:
            green_ene = self.green_trace.get(inv.region_id, {}).get(int(inv.timestamp // 3600 * 3600))
            if green_ene is None:
                print("No energy data provided.")
                continue

            self.total_invocations += 1
            is_cold = (
                inv.pod_id not in self.container_state or
                inv.timestamp > self.container_state[inv.pod_id][0] + self.fixed_keep_alive_s
            )

            net_latency = 2 * self.network_latency_map.get((inv.user_region, inv.region_id), 0.0)
            if is_cold:
                self.cold_start_count += 1
                self.total_latency += inv.exec_time_s / 0.9 + inv.cold_start_latency_s + net_latency
            else:
                self.total_latency += inv.exec_time_s / 0.9 + net_latency

            self.total_network_latency += net_latency

            e_energy = self.energy_estimator.estimate_exec_energy(inv.exec_time_s, inv.cpu_cores, inv.mem_MB)
            e_carbon = self.energy_estimator.estimate_carbon_emission(e_energy, green_ene.carbon_intensity)
            self.execution_energy += e_energy
            self.execution_carbon += e_carbon

            if inv.pod_id in self.container_state:
                last_ts, mem_MB, cpu_cores = self.container_state[inv.pod_id]
                idle_time = inv.timestamp - last_ts
                if idle_time > 0:
                    recorded_idle_time = min(idle_time, self.fixed_keep_alive_s)
                    k_energy, k_carbon = self.energy_estimator.estimate_keep_alive_energy(
                        mem_MB, cpu_cores, recorded_idle_time, green_ene.carbon_intensity
                    )
                    self.keep_alive_energy += k_energy
                    self.keep_alive_carbon += k_carbon
                    self.total_idle_time.setdefault(inv.pod_id, 0.0)
                    self.total_idle_time[inv.pod_id] += recorded_idle_time
                    self.total_idle_energy.setdefault(inv.pod_id, 0.0)
                    self.total_idle_energy[inv.pod_id] += k_energy

            self.container_state[inv.pod_id] = (
                inv.timestamp,
                inv.mem_MB,
                inv.cpu_cores
            )

        self.total_energy = self.execution_energy + self.keep_alive_energy
        self.total_carbon = self.execution_carbon + self.keep_alive_carbon

        return {
            'total_energy_J': self.total_energy,
            'total_carbon_g': self.total_carbon,
            'cold_starts': self.cold_start_count,
            'avg_latency_s': self.total_latency / self.total_invocations if self.total_invocations else 0.0,
            'total_network_latency_s': self.total_network_latency,
            'invocation_count': self.total_invocations,
            'keep_alive_energy_J': self.keep_alive_energy,
            'execution_energy_J': self.execution_energy,
            'keep_alive_carbon_g': self.keep_alive_carbon,
            'execution_carbon_g': self.execution_carbon,
            'idle_time_by_function': self.total_idle_time,
            'idle_energy_by_function': self.total_idle_energy
        }

    def run_with_agent(self, invocations: List[FunctionInvocation], agent, obs_fn):
            self.reset_metrics()
            for inv in invocations:
                green_ene = self.green_trace.get(inv.region_id, {}).get(int(inv.timestamp // 3600 * 3600))
                if green_ene is None:
                    continue

                obs = obs_fn(inv)
                action = agent.select_action(obs)
                # dest_region, keep_alive_s = agent.action_lookup[action]
                _, keep_alive_s = agent.action_lookup[action]
                dest_region = inv.user_region # Hold for future cross-region usage

                self.total_invocations += 1
                is_cold = (
                    inv.pod_id not in self.container_state or
                    inv.timestamp > self.container_state[inv.pod_id][0] + keep_alive_s
                )

                net_latency = 2 * self.network_latency_map.get((inv.user_region, dest_region), 0.0)
                if is_cold:
                    self.cold_start_count += 1
                    self.total_latency += inv.exec_time_s + inv.cold_start_latency_s + net_latency
                else:
                    self.total_latency += inv.exec_time_s + net_latency

                self.total_network_latency += net_latency

                e_energy = self.energy_estimator.estimate_exec_energy(inv.exec_time_s, inv.cpu_cores, inv.mem_MB)
                e_carbon = self.energy_estimator.estimate_carbon_emission(e_energy, green_ene.carbon_intensity)
                self.execution_energy += e_energy
                self.execution_carbon += e_carbon

                if inv.pod_id in self.container_state:
                    last_ts, mem_MB, cpu_cores = self.container_state[inv.pod_id]
                    idle_time = inv.timestamp - last_ts
                    if idle_time > 0:
                        recorded_idle_time = min(idle_time, keep_alive_s)
                        k_energy, k_carbon = self.energy_estimator.estimate_keep_alive_energy(
                            mem_MB, cpu_cores, recorded_idle_time, green_ene.carbon_intensity
                        )
                        self.keep_alive_energy += k_energy
                        self.keep_alive_carbon += k_carbon
                        self.total_idle_time.setdefault(inv.pod_id, 0.0)
                        self.total_idle_time[inv.pod_id] += recorded_idle_time
                        self.total_idle_energy.setdefault(inv.pod_id, 0.0)
                        self.total_idle_energy[inv.pod_id] += k_energy

                self.container_state[inv.pod_id] = (
                    inv.timestamp,
                    inv.mem_MB,
                    inv.cpu_cores
                )

            self.total_energy = self.execution_energy + self.keep_alive_energy
            self.total_carbon = self.execution_carbon + self.keep_alive_carbon

            return {
                'total_energy_J': self.total_energy,
                'total_carbon_g': self.total_carbon,
                'cold_starts': self.cold_start_count,
                'avg_latency_s': self.total_latency / self.total_invocations if self.total_invocations else 0.0,
                'total_network_latency_s': self.total_network_latency,
                'invocation_count': self.total_invocations,
                'keep_alive_energy_J': self.keep_alive_energy,
                'execution_energy_J': self.execution_energy,
                'keep_alive_carbon_g': self.keep_alive_carbon,
                'execution_carbon_g': self.execution_carbon,
                'idle_time_by_function': self.total_idle_time,
                'idle_energy_by_function': self.total_idle_energy
            }

    def run_with_agent_with_action_log(self, invocations: List[FunctionInvocation], agent, obs_fn):
            self.reset_metrics()
            actions = []
            for inv in invocations:
                green_ene = self.green_trace.get(inv.region_id, {}).get(int(inv.timestamp // 3600 * 3600))
                if green_ene is None:
                    continue

                obs = obs_fn(inv)
                action = agent.select_action(obs)
                # dest_region, keep_alive_s = agent.action_lookup[action]
                _, keep_alive_s = agent.action_lookup[action]
                actions.append(keep_alive_s)
                dest_region = inv.user_region

                self.total_invocations += 1
                is_cold = (
                    inv.pod_id not in self.container_state or
                    inv.timestamp > self.container_state[inv.pod_id][0] + keep_alive_s
                )

                net_latency = 2 * self.network_latency_map.get((inv.user_region, dest_region), 0.0)
                if is_cold:
                    self.cold_start_count += 1
                    self.total_latency += inv.exec_time_s + inv.cold_start_latency_s + net_latency
                else:
                    self.total_latency += inv.exec_time_s + net_latency

                self.total_network_latency += net_latency

                e_energy = self.energy_estimator.estimate_exec_energy(inv.exec_time_s, inv.cpu_cores, inv.mem_MB)
                e_carbon = self.energy_estimator.estimate_carbon_emission(e_energy, green_ene.carbon_intensity)
                self.execution_energy += e_energy
                self.execution_carbon += e_carbon

                if inv.pod_id in self.container_state:
                    last_ts, mem_MB, cpu_cores = self.container_state[inv.pod_id]
                    idle_time = inv.timestamp - last_ts
                    if idle_time > 0:
                        recorded_idle_time = min(idle_time, keep_alive_s)
                        k_energy, k_carbon = self.energy_estimator.estimate_keep_alive_energy(
                            mem_MB, cpu_cores, recorded_idle_time, green_ene.carbon_intensity
                        )
                        self.keep_alive_energy += k_energy
                        self.keep_alive_carbon += k_carbon
                        self.total_idle_time.setdefault(inv.pod_id, 0.0)
                        self.total_idle_time[inv.pod_id] += recorded_idle_time
                        self.total_idle_energy.setdefault(inv.pod_id, 0.0)
                        self.total_idle_energy[inv.pod_id] += k_energy

                self.container_state[inv.pod_id] = (
                    inv.timestamp,
                    inv.mem_MB,
                    inv.cpu_cores
                )

            self.total_energy = self.execution_energy + self.keep_alive_energy
            self.total_carbon = self.execution_carbon + self.keep_alive_carbon

            return {
                'total_energy_J': self.total_energy,
                'total_carbon_g': self.total_carbon,
                'cold_starts': self.cold_start_count,
                'avg_latency_s': self.total_latency / self.total_invocations if self.total_invocations else 0.0,
                'total_network_latency_s': self.total_network_latency,
                'invocation_count': self.total_invocations,
                'keep_alive_energy_J': self.keep_alive_energy,
                'execution_energy_J': self.execution_energy,
                'keep_alive_carbon_g': self.keep_alive_carbon,
                'execution_carbon_g': self.execution_carbon,
                'idle_time_by_function': self.total_idle_time,
                'idle_energy_by_function': self.total_idle_energy
            }, actions
