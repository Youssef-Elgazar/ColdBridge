import json
import argparse
import time
import torch
import os

from lace_rl.agent.dqn_agent import DQNAgent
from lace_rl.sim.trace_simulator import TraceDrivenSimulator
from lace_rl.utils.io import load_demo_inputs, ensure_dir

def build_action_lookup(keep_alive_options, regions):
    return [(region, t) for region in regions for t in keep_alive_options]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    # agent config
    agent = DQNAgent(cfg["agent"]["obs_dim"], cfg["agent"]["n_actions"])
    checkpoint = cfg["agent"]["checkpoint"]
    agent.q_network.load_state_dict(torch.load(checkpoint))
    agent.q_network.eval()

    agent.action_lookup = build_action_lookup(
        cfg["policy"]["keep_alive_options"],
        cfg["policy"]["regions"],
    )

    # inputs
    invocations, green_trace, network_latency_map, obs_fn = load_demo_inputs(cfg)

    # simulator
    sim = TraceDrivenSimulator(
        green_trace=green_trace,
        network_latency_map=network_latency_map,
    )

    lam = cfg["eval"]["lambda_carbon"]
    t0 = time.time()
    results = sim.run_with_agent(invocations, agent, lambda inv: obs_fn(inv, lambda_carbon=lam))
    t1 = time.time()

    ensure_dir(cfg["eval"]["out_dir"])
    out_path = f'{cfg["eval"]["out_dir"]}/metrics.json'
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Output {out_path}")
    print(f"Execution time: {t1 - t0:.2f}s")

if __name__ == "__main__":
    main()
