"""kvpipe CLI: doctor | calibrate | run | report | audit | regress."""

from __future__ import annotations

import json

import click
from rich.console import Console
from rich.table import Table

from kvpipe.doctor import run_doctor

console = Console()


@click.group()
def main() -> None:
    """kvpipe: KV-Cache Compression Pipeline and Evaluation Toolkit."""


@main.command()
@click.option("--env-file", type=click.Path(exists=True), default=None, help="Path to cached env.json")
@click.option("--json", "json_output", is_flag=True, default=False, help="Output as JSON")
def doctor(env_file: str | None, json_output: bool) -> None:
    """Hardware and engine discovery, tier selection, and arm support check."""
    report = run_doctor(env_file)
    if json_output:
        click.echo(json.dumps(report, indent=2))
        return

    console.print(f"[bold cyan]kvpipe doctor[/bold cyan] - Hardware Tier: [bold green]{report['tier']}[/bold green]")
    console.print(f"Python: {report['python']} | GPU Available: {report['gpu_available']}")

    if report["gpus"]:
        gpu_table = Table(title="Detected GPUs")
        gpu_table.add_column("ID", justify="right")
        gpu_table.add_column("Name")
        gpu_table.add_column("Compute Cap")
        gpu_table.add_column("Memory (MB)")
        for g in report["gpus"]:
            gpu_table.add_row(str(g.get("id", 0)), g.get("name", "Unknown"), str(g.get("compute_cap", "")), str(g.get("total_memory_mb", "")))
        console.print(gpu_table)
    else:
        console.print("[yellow]No NVIDIA GPU detected locally. Arms requiring GPU engines will be disabled or run in stub mode.[/yellow]")

    pkg_table = Table(title="Inference Engines & Packages")
    pkg_table.add_column("Package")
    pkg_table.add_column("Installed Version")
    for pkg, ver in report["packages"].items():
        pkg_table.add_row(pkg, ver if ver else "[red]Not installed[/red]")
    console.print(pkg_table)

    arm_table = Table(title="Arm Candidate Evaluation")
    arm_table.add_column("Arm ID")
    arm_table.add_column("Role")
    arm_table.add_column("Engine")
    arm_table.add_column("KV Dtype")
    arm_table.add_column("Supported?", justify="center")
    arm_table.add_column("Reason / Notes")

    for a in report["arms"]:
        status_str = "[bold green]YES[/bold green]" if a["supported"] else "[bold red]NO[/bold red]"
        arm_table.add_row(a["id"], a["role"], a["engine"], a["kv_dtype"], status_str, a["reason"])
    console.print(arm_table)


if __name__ == "__main__":
    main()
