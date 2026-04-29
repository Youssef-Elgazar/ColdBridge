import os
import pickle
from typing import Any, Dict, List, Tuple

from lace_rl.sim.trace_simulator import FunctionInvocation, GreenEnergy

class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "SimulationwithLatency":
            module = "lace_rl.sim.trace_simulator"
        return super().find_class(module, name)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_pickle(path: str):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError:
            f.seek(0)
            return _CompatUnpickler(f).load()

def _to_invocation(x: Any) -> FunctionInvocation:
    if isinstance(x, FunctionInvocation):
        return x
    if isinstance(x, dict):
        return FunctionInvocation(**x)
    
    return FunctionInvocation(
        timestamp=int(getattr(x, "timestamp")),
        pod_id=str(getattr(x, "pod_id")),
        region_id=str(getattr(x, "region_id")),
        exec_time_s=float(getattr(x, "exec_time_s")),
        cpu_cores=float(getattr(x, "cpu_cores")),
        mem_MB=float(getattr(x, "mem_MB")),
        cold_start_latency_s=float(getattr(x, "cold_start_latency_s", 0.0) or 0.0),
        user_region=str(getattr(x, "user_region", "local")),
        metadata=getattr(x, "metadata", None),
    )


def _to_green_trace(obj: Any) -> Dict[str, Dict[int, GreenEnergy]]:
    out: Dict[str, Dict[int, GreenEnergy]] = {}
    for region, hour_map in obj.items():
        out[region] = {}
        for hour_ts, v in hour_map.items():
            hour_ts = int(hour_ts)
            if isinstance(v, GreenEnergy):
                out[region][hour_ts] = v
            elif isinstance(v, (int, float)):
                out[region][hour_ts] = GreenEnergy(timestamp=hour_ts, region_id=str(region), carbon_intensity=float(v))
            else:
                ci = float(getattr(v, "carbon_intensity"))
                out[region][hour_ts] = GreenEnergy(timestamp=hour_ts, region_id=str(region), carbon_intensity=ci)
    return out


def _apply_cs_latency(inv: FunctionInvocation, cs_dict: Any) -> FunctionInvocation:
    if inv.cold_start_latency_s and inv.cold_start_latency_s > 0:
        return inv

    if isinstance(cs_dict, dict):
        if inv.pod_id in cs_dict:
            inv.cold_start_latency_s = float(cs_dict[inv.pod_id])
            return inv
        if "default" in cs_dict:
            inv.cold_start_latency_s = float(cs_dict["default"])
            return inv
    return inv


def load_demo_inputs(cfg: dict):
    inv_path = cfg["inputs"]["invocations"]
    green_path = cfg["inputs"]["green_trace"]
    net_path = cfg["inputs"]["network_latency_map"]
    cs_path = cfg["inputs"].get("cs_latency_dict")

    raw_inv = _load_pickle(inv_path)
    raw_green = _load_pickle(green_path)
    network_latency_map = _load_pickle(net_path)

    cs_dict = _load_pickle(cs_path) if cs_path else None

    invocations: List[FunctionInvocation] = [_to_invocation(x) for x in raw_inv]
    if cs_dict is not None:
        invocations = [_apply_cs_latency(inv, cs_dict) for inv in invocations]

    green_trace = _to_green_trace(raw_green)

    def obs_fn(inv: FunctionInvocation, lambda_carbon: float = 0.1):
        hour = int(inv.timestamp // 3600 * 3600)
        ci = green_trace.get(inv.region_id, {}).get(hour, GreenEnergy(hour, inv.region_id, 0.0)).carbon_intensity
        net = 2 * network_latency_map.get((inv.user_region, inv.region_id), 0.0)

        return [
            float(inv.cpu_cores),                 
            float(inv.mem_MB),                    
            float(inv.exec_time_s),              
            float(inv.cold_start_latency_s),      
            float(ci),
            float(lambda_carbon),
            float(net),
            0.0, 0.0, 0.0, 0.0  # reserved
        ]

    return invocations, green_trace, network_latency_map, obs_fn
