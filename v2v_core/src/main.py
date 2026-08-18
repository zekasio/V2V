"""
V2V - AI Video Localization CLI

Commands:
  python -m src.main process <video>    Process a single video
  python -m src.main watch              Watch input/ folder for new videos
  python -m src.main config-test        Validate configuration
"""
from __future__ import annotations
import asyncio
import os
import sys
from pathlib import Path

# Fix Windows console encoding for Unicode
if sys.platform == "win32":
    os.system("")  # Enable ANSI escape codes on Windows
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.config import settings
from src.logger import setup_logging, get_logger

console = Console(force_terminal=True)
logger = get_logger("main")


@click.group()
@click.version_option("1.0.0", prog_name="V2V")
def cli():
    """V2V - AI Video Localization Pipeline (TR > EN)"""
    setup_logging()


@cli.command()
@click.argument("video", type=click.Path(exists=True))
@click.option("--subtitle-mode", type=click.Choice(["external", "burned"]), default=None,
              help="Override subtitle mode")
@click.option("--fast-mode", is_flag=True, help="Enable ultra-fast static template mode")
def process(video: str, subtitle_mode: str | None, fast_mode: bool):
    """Process a single video file."""
    from src.pipeline import LocalizationPipeline

    video_path = Path(video).resolve()
    console.print(Panel(
        f"[bold cyan]Processing:[/] {video_path.name}\n"
        f"[dim]Fast Mode: {fast_mode}[/]",
        title="[V2V Localization]",
        border_style="cyan",
    ))

    if subtitle_mode:
        settings.subtitle_mode = subtitle_mode
    if fast_mode:
        settings.fast_template_mode = True

    pipeline = LocalizationPipeline()

    try:
        output = asyncio.run(pipeline.process_video(video_path))
        console.print(Panel(
            f"[bold green]Output:[/] {output}",
            title="Complete",
            border_style="green",
        ))
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled by user[/]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        logger.exception("Pipeline failed")
        sys.exit(1)


@cli.command()
def watch():
    """Watch input/ folder for new video files and process automatically."""
    from src.pipeline import LocalizationPipeline
    from src.watcher import FolderWatcher

    console.print(Panel(
        f"[bold cyan]Watching:[/] {settings.input_path}\n"
        f"[dim]Drop .mp4 / .mov files to auto-process[/]",
        title="[V2V Watcher]",
        border_style="cyan",
    ))

    pipeline = LocalizationPipeline()

    async def on_video(path: Path):
        await pipeline.process_video(path)

    watcher = FolderWatcher(on_video)

    try:
        asyncio.run(watcher.start())
    except KeyboardInterrupt:
        console.print("\n[yellow]Watcher stopped[/]")
        watcher.stop()


@cli.command("config-test")
def config_test():
    """Validate configuration and check dependencies."""
    console.print(Panel("[bold]Configuration Check[/]", title="[V2V Config]", border_style="cyan"))

    table = Table(title="Settings", show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")
    table.add_column("Status", style="green")

    # API Keys
    murf_ok = bool(settings.murfdub_api_key and settings.murfdub_api_key != "your_murfdub_api_key_here")
    table.add_row("Murf API Key", "***" + settings.murfdub_api_key[-4:] if murf_ok else "NOT SET",
                  "[green]OK[/]" if murf_ok else "[red]MISSING[/]")

    gemini_ok = bool(settings.gemini_api_key and settings.gemini_api_key != "your_gemini_api_key_here")
    table.add_row("Gemini API Key", "***" + settings.gemini_api_key[-4:] if gemini_ok else "NOT SET",
                  "[green]OK[/]" if gemini_ok else "[red]MISSING[/]")

    table.add_row("LLM Model", settings.gemini_model, "[green]OK[/]")
    table.add_row("Target Locale", settings.target_locale, "[green]OK[/]")
    table.add_row("Subtitle Mode", settings.subtitle_mode, "[green]OK[/]")
    table.add_row("Inpainting", settings.inpainting_method, "[green]OK[/]")

    # Directories
    for name, path in [("Input", settings.input_path), ("Output", settings.output_path),
                       ("Temp", settings.temp_path), ("Logs", settings.log_path)]:
        table.add_row(f"{name} Dir", str(path), "[green]OK[/]" if path.exists() else "[yellow]Created[/]")

    console.print(table)

    # Check dependencies
    console.print("\n[bold]Dependencies:[/]")
    deps = [
        ("ffmpeg", _check_cmd("ffmpeg -version")),
        ("OpenCV", _check_import("cv2")),
        ("Pillow", _check_import("PIL")),
        ("Murf SDK", _check_import("murf")),
        ("Google GenAI", _check_import("google.genai")),
    ]
    optional_deps = [
        ("PyTorch (Optional for LaMa)", _check_import("torch")),
    ]
    for name, ok in deps:
        status = "[green]OK[/]" if ok else "[red]MISSING[/]"
        console.print(f"  {status}  {name}")
    for name, ok in optional_deps:
        status = "[green]OK[/]" if ok else "[yellow]OPTIONAL (MISSING)[/]"
        console.print(f"  {status}  {name}")

    all_ok = murf_ok and gemini_ok and all(ok for _, ok in deps)
    if all_ok:
        console.print(Panel("[bold green]All checks passed![/]", border_style="green"))
    else:
        console.print(Panel("[bold yellow]Some checks failed - review above[/]", border_style="yellow"))


def _check_import(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def _check_cmd(cmd: str) -> bool:
    import subprocess
    try:
        subprocess.run(cmd.split(), capture_output=True, timeout=5)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    cli()
