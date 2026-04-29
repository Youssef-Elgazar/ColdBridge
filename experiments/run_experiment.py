"""
run_experiment.py — Main CLI entry point
=========================================

Examples:
  # Baseline: measure raw cold starts with no modules
  python -m experiments.run_experiment --mode baseline --invocations 30

  # Module A only: prediction + pre-warming
  python -m experiments.run_experiment --mode module_a --invocations 30

  # Full suite (B and C skipped until teammates deliver)
  python -m experiments.run_experiment --mode full --skip_b --skip_c --invocations 30

  # Realistic timing (paces invocations to match trace inter-arrival times)
  python -m experiments.run_experiment --mode module_a --realtime --duration 300

  # Compare all saved runs
  python -m experiments.run_experiment --compare
"""

import logging
import sys
import time
from pathlib import Path

import click
from rich.console import Console

# Make sure project root is on PYTHONPATH when running as module
sys.path.insert(0, str(Path(__file__).parent.parent))

from coldbridge.worker.pool import DockerWorkerPool, DEFAULT_FUNCTIONS, FunctionSpec
from coldbridge.modules.module_a import ModuleA
from coldbridge.modules.module_b import ModuleB
from coldbridge.modules.module_c import ModuleC
from coldbridge.modules.module_d import ModuleD, compare_runs
from coldbridge.data.trace_loader import SyntheticTraceGenerator
from coldbridge.data.real_trace_loaders import (
    Industry40ColdStartLoader,
    ZenodoHuaweiLoader,
    AzureFunctions2019Loader,
)

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
# Quieten noisy libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("docker").setLevel(logging.WARNING)


def _function_filter(all_fns, names_str: str):
    """Filter DEFAULT_FUNCTIONS by comma-separated names string."""
    if names_str == "all":
        return all_fns
    selected = set(names_str.split(","))
    return [f for f in all_fns if f.name in selected]


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command()
@click.option("--mode", default="baseline",
              type=click.Choice(["baseline", "module_a", "module_b", "module_c", "full"]),
              help="Which modules to activate.")
@click.option("--functions", default="all",
              help="Comma-separated function names to include: python_fn,node_fn,java_fn")
@click.option("--invocations", default=30, type=int,
              help="Approximate total invocations across all functions.")
@click.option("--duration", default=None, type=float,
              help="Trace duration in seconds (overrides --invocations).")
@click.option("--realtime", is_flag=True, default=False,
              help="Pace invocations to real trace timing (slow but more realistic).")
@click.option("--trace", "trace_source", default="synthetic",
              type=click.Choice(["synthetic", "industry40", "zenodo", "azure"]),
              help="Trace data source: synthetic, industry40, zenodo (Huawei), or azure.")
@click.option("--max-trace-events", "max_trace_events", default=3000, type=int,
              help="Max events to load from real traces.")
@click.option("--skip_b", is_flag=True, default=False, help="Disable Module B.")
@click.option("--skip_c", is_flag=True, default=False, help="Disable Module C.")
@click.option("--theta", default=0.45, type=float,
              help="Module A pre-warm decision threshold.")
@click.option("--ttl", default=120.0, type=float,
              help="Container keep-alive TTL in seconds.")
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.option("--out", default=None, type=str,
              help="Output directory (default: results/<timestamp>_<mode>).")
def run(mode, functions, invocations, duration, realtime, trace_source,
        max_trace_events, skip_b, skip_c, theta, ttl, seed, out):
    """Run a single experiment and save results."""

    # ── Output directory ──────────────────────────────────────────────────
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(out) if out else Path("results") / f"{ts}_{mode}_{trace_source}"

    # ── Function specs ────────────────────────────────────────────────────
    fn_specs = _function_filter(DEFAULT_FUNCTIONS, functions)
    for spec in fn_specs:
        spec.ttl_seconds = ttl
    if not fn_specs:
        console.print("[red]No matching function specs. Check --functions.[/red]")
        return

    # ── Load or generate trace ────────────────────────────────────────────
    events = None

    if trace_source == "industry40":
        loader = Industry40ColdStartLoader()
        events = loader.load(max_events=max_trace_events)
        console.print(f"\n[bold cyan]Trace source:[/bold cyan] Industry 4.0 Cold Start (IEEE JSAS 2024)")

    elif trace_source == "zenodo":
        loader = ZenodoHuaweiLoader()
        events = loader.load(max_events=max_trace_events)
        console.print(f"\n[bold cyan]Trace source:[/bold cyan] Zenodo/Huawei LACE-RL (CCGrid 2026)")

    elif trace_source == "azure":
        loader = AzureFunctions2019Loader()
        events = loader.load(max_events=max_trace_events)
        console.print(f"\n[bold cyan]Trace source:[/bold cyan] Azure Functions 2019 (ATC 2020)")

    else:
        # Synthetic trace (original behaviour)
        gen = SyntheticTraceGenerator(seed=seed)
        fn_names = [f.name for f in fn_specs]
        if duration:
            dur = duration
            configs = [
                {"name": n, "pattern": _pattern_for(n), "mean_iat_s": _iat_for(n)}
                for n in fn_names
            ]
        else:
            avg_iat = 30.0
            dur = invocations * avg_iat / max(len(fn_specs), 1)
            configs = [
                {"name": n, "pattern": _pattern_for(n),
                 "mean_iat_s": dur / max(invocations // len(fn_specs), 1)}
                for n in fn_names
            ]
        events = gen.generate(duration_seconds=dur, function_configs=configs)
        console.print(f"\n[bold cyan]Trace source:[/bold cyan] Synthetic (seed={seed})")

    if not events:
        console.print("[red]No events loaded. Check your trace files or try increasing --invocations.[/red]")
        return

    trace_fns = sorted({e.function_name for e in events})
    
    if trace_source != "synthetic":
        # Generate FunctionSpec for each unique function in the real trace
        # Map them all to the lightweight python_fn image
        fn_specs = []
        for f_name in trace_fns:
            fn_specs.append(FunctionSpec(
                name=f_name,
                image="coldbridge/python-fn:latest",
                port=8080,
                runtime="python",
                ttl_seconds=ttl,
            ))

    console.print(f"[bold]Mode:[/bold] {mode}  |  "
                  f"[bold]Functions:[/bold] {trace_fns[:5]}... ({len(trace_fns)} total)  |  "
                  f"[bold]Events:[/bold] {len(events)}\n")

    # ── Build worker pool ─────────────────────────────────────────────────
    pool = DockerWorkerPool(fn_specs)

    try:
        # ── Build modules ─────────────────────────────────────────────────
        mod_a = None
        mod_b = None
        mod_c = None

        if mode in ("module_a", "full"):
            mod_a = ModuleA(pool, theta=theta)

        if mode in ("module_b", "full") and not skip_b:
            mod_b = ModuleB()

        if mode in ("module_c", "full") and not skip_c:
            mod_c = ModuleC()

        # ── Run ───────────────────────────────────────────────────────────
        harness = ModuleD(
            worker_pool=pool,
            mode=mode,
            module_a=mod_a,
            module_b=mod_b,
            module_c=mod_c,
            inter_event_realtime=realtime,
            ttl_seconds=ttl,
        )
        metrics = harness.run(events)
        harness.print_summary(metrics)
        harness.save_results(out_dir)

        # If Module A is active, log model parameter count
        if mod_a:
            console.print(
                f"\n[dim]Module A parameters: {mod_a.parameter_count():,}[/dim]"
            )

    finally:
        pool.shutdown()


@cli.command()
@click.option("--results_dir", default="results", type=str,
              help="Directory containing run sub-folders.")
def compare(results_dir):
    """Compare all saved experiment runs."""
    compare_runs(Path(results_dir))


# ── helpers ───────────────────────────────────────────────────────────────────

def _pattern_for(fn_name: str) -> str:
    return {"python_fn": "periodic", "node_fn": "bursty", "java_fn": "rare"}.get(fn_name, "periodic")

def _iat_for(fn_name: str) -> float:
    return {"python_fn": 30.0, "node_fn": 45.0, "java_fn": 90.0}.get(fn_name, 30.0)


if __name__ == "__main__":
    cli()
