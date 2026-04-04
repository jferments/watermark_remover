#!/usr/bin/env python3
"""
watermark_remover.py — DETECT (fancyfeast YOLOv11 finetune) + ERASE (Simple-LaMa)
======================================================================

Multi-GPU enabled fast batch removal of watermarks from large image datasets

Key Architectural Features:
- Multi GPU (AI) and CPU (I/O) workloads for true parallelism and high throughput.
- Resumable processing via a `.processing_log.txt` checkpoint file.
- Per-GPU status tracking in a pretty Rich console display

--------------------------------------------------------------------
USAGE INSTRUCTIONS
--------------------------------------------------------------------

   python3 watermark_remover.py -i /path/to/inputs -o /path/to/outputs -R

--------------------------------------------------------------------

 ╭────────────────────────────────────────────────────────────────────────╮
 │                  DISCLAIMER & RESPONSIBLE USE NOTICE                   │
 ╰────────────────────────────────────────────────────────────────────────╯
 This script is provided for educational and technical demonstration
 purposes only. Removing watermarks from images may violate copyright
 or intellectual property rights. Users of this script are solely
 responsible for ensuring they have the legal right to modify the images
 they process. The author assumes no liability for misuse of this tool.

"""

# ─────────────────────────────────── Standard Library Imports ───────────────────────────────────
import os
import sys
import argparse
import pathlib
import signal
import time
from collections import deque
import multiprocessing as mp
import subprocess
import queue # For queue.Empty exception

# ────────────────────────────────── Third-Party Library Imports ─────────────────────────────────
try:
    from rich.console import Console, Group
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
except ImportError:
    print("Error: The 'rich' library is not found. Please install it with 'pip install rich'")
    sys.exit(1)

# ───────────────────────────────────────────────────────────────────────────────────────────────

IMG_EXTS = ("jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff")

# ╭───────────────────────── CLI Argument Parsing ─────────────────────────╮
def parse_cli_args():
    """Sets up and parses command-line arguments using Python's argparse."""
    parser = argparse.ArgumentParser(
        description="Detect watermarks with a custom YOLOv11 model and in-paint with Simple-LaMa.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-i", "--input", required=True, type=pathlib.Path, help="Path to the folder containing watermarked images.")
    parser.add_argument("-o", "--output", required=True, type=pathlib.Path, help="Path to the folder where clean images will be saved.")
    parser.add_argument("-w", "--weights", type=pathlib.Path, default=pathlib.Path("yolo11x-train28-best.pt"), help="Path to the YOLOv11 model weights file.")
    parser.add_argument("--conf", type=float, default=0.1, help="YOLO detection confidence threshold.")
    parser.add_argument("--dilate", type=int, default=15, help="Pixel amount to expand detected masks.")
    parser.add_argument("-R", "--recursive", action="store_true", help="Process images in subdirectories recursively.")
    parser.add_argument("--cpu-workers", type=int, default=os.cpu_count(), help="Total number of CPU processes for writing images to disk.")
    parser.add_argument("--debug", action="store_true", help="Save intermediate mask_raw and mask_preview images for debugging.")
    return parser.parse_args()
# ╰─────────────────────────────────────────────────────────────────────────╯

# ╭─────────────────── System, Checkpoint & File Discovery Functions ───────────────────╮
def get_gpu_ids():
    """Detects available NVIDIA GPU IDs by shelling out to `nvidia-smi`."""
    try:
        smi_output = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], encoding="utf-8").strip()
        return [int(line) for line in smi_output.splitlines()]
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return []

def load_processed_files(log_file_path: pathlib.Path) -> set:
    """Loads the set of already processed file paths from the checkpoint log."""
    if not log_file_path.exists(): return set()
    with open(log_file_path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f)

def find_image_files(input_dir: pathlib.Path, recursive: bool, console: Console):
    """Scans for image files, displaying a live counter of the scan."""
    console.print(f"[*] Scanning for images in: [cyan]{input_dir}[/cyan]")
    image_files = []
    with console.status("[bold green]Searching for images...") as status:
        if recursive:
            dirs_scanned = 0
            for dirpath, _, filenames in os.walk(input_dir):
                dirs_scanned += 1
                for filename in filenames:
                    if filename.lower().endswith(IMG_EXTS):
                        image_files.append(pathlib.Path(dirpath) / filename)
                status.update(f"[bold green]Scanning... Dirs: {dirs_scanned:,} | Images found: {len(image_files):,}")
        else:
            all_items = list(input_dir.iterdir())
            for i, item in enumerate(all_items):
                if item.is_file() and item.suffix.lower().lstrip('.') in IMG_EXTS:
                    image_files.append(item)
                status.update(f"[bold green]Scanning... Items: {i+1}/{len(all_items)} | Images found: {len(image_files):,}")
    return image_files
# ╰─────────────────────────────────────────────────────────────────────────╯

# ╭─────────────────────── Custom Rich Progress Columns ───────────────────────╮
class SpeedColumn(TextColumn):
    """A custom `rich` progress column for displaying speed in images/sec."""
    def __init__(self, *args, **kwargs): super().__init__(" ", *args, **kwargs)
    def render(self, task) -> Text:
        if task.speed is None: return Text("N/A", style="dim green")
        return Text(f"{task.speed:.2f} img/s", style="green")

class EstimatedTimeRemainingColumn(TimeRemainingColumn):
    """A `TimeRemainingColumn` that adds a custom label."""
    def render(self, task) -> Text:
        if task.finished or task.time_remaining is None: return Text("-:--:--", style="dim")
        return Text("Est. time remaining: ", style="dim") + super().render(task)
# ╰─────────────────────────────────────────────────────────────────────────╯

# ╭─────────────────────── GPU & CPU Worker Implementations ───────────────────────╮
def gpu_worker_process(gpu_id: int, image_paths: list, write_queue: mp.Queue, status_queue: mp.Queue, args: argparse.Namespace):
    """The core function executed by each GPU worker process."""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch, cv2, numpy as np
    from ultralytics import YOLO
    from simple_lama_inpainting import SimpleLama
    from PIL import Image
    try:
        device = torch.device("cuda:0"); yolo_model = YOLO(args.weights).to(device); lama_model = SimpleLama(device=device)
    except Exception as e:
        status_queue.put({"type": "error", "message": f"GPU {gpu_id} failed to init: {e}"}); return

    for path in image_paths:
        try:
            img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                status_queue.put({"type": "log", "message": f"Could not read image: {path}"}); continue
            predictions = yolo_model(img_bgr, conf=args.conf, verbose=False)[0]
            if len(predictions.boxes.xyxy) > 0:
                mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
                for box in predictions.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box); cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
                if args.dilate > 0:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.dilate, args.dilate)); mask = cv2.dilate(mask, kernel, iterations=1)
                if args.debug:
                    preview_overlay = np.zeros_like(img_bgr); preview_overlay[mask == 255] = [0, 0, 255]
                    mask_preview = cv2.addWeighted(img_bgr, 0.7, preview_overlay, 0.3, 0)
                    relative_path = path.relative_to(args.input)
                    debug_dir = args.output / "debug" / relative_path.parent
                    write_queue.put((debug_dir / f"{path.stem}_mask_raw.png", mask))
                    write_queue.put((debug_dir / f"{path.stem}_mask_preview.png", mask_preview))
                result_bgr = cv2.cvtColor(np.array(lama_model(Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)), Image.fromarray(mask))), cv2.COLOR_RGB2BGR)
            else:
                result_bgr = img_bgr
            write_queue.put((args.output / path.relative_to(args.input), result_bgr))
            status_queue.put({"type": "gpu_progress", "gpu_id": gpu_id})
        except Exception as e:
            status_queue.put({"type": "error", "message": f"Error on GPU {gpu_id} processing {path.name}: {e}"})

def cpu_writer_process(write_queue: mp.Queue, log_queue: mp.Queue, output_dir: pathlib.Path):
    """A dedicated I/O process that saves files and reports success for checkpointing."""
    import cv2
    created_dirs = set()
    while True:
        try:
            item = write_queue.get()
            if item is None: break
            path, image_array = item
            if path.parent not in created_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)
                created_dirs.add(path.parent)
            success, buffer = cv2.imencode(path.suffix, image_array)
            if success:
                buffer.tofile(str(path))
            else:
                print(f"[!] Failed to save image {path.name}")
            if not str(path).startswith(str(output_dir / "debug")):
                 log_queue.put(path.relative_to(output_dir).as_posix())
        except (KeyboardInterrupt, SystemExit):
            break
        except Exception: continue
# ╰─────────────────────────────────────────────────────────────────────────╯

# ╭────────────────────────────── Main Driver ──────────────────────────────╮
def main():
    """The main process, orchestrator of all workers and the UI."""
    console = Console()
    args = parse_cli_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.debug: (args.output / "debug").mkdir(parents=True, exist_ok=True)

    # --- Checkpoint and File Discovery ---
    log_file_path = args.output / ".processing_log.txt"
    processed_files_set = load_processed_files(log_file_path)
    if processed_files_set:
        console.print(f"[*] Found checkpoint file. Resuming session. [bold yellow]{len(processed_files_set):,}[/] files will be skipped.")

    all_image_paths = find_image_files(args.input, args.recursive, console)
    images_to_process = [p for p in all_image_paths if p.relative_to(args.input).as_posix() not in processed_files_set]

    if not images_to_process:
        console.print("[bold green]✅ All images have already been processed. Nothing to do.[/bold green]"); sys.exit(0)

    console.print(f"[*] Total images in dataset: {len(all_image_paths):,}. New images to process this session: [bold green]{len(images_to_process):,}[/bold green]")

    gpu_ids = get_gpu_ids()
    if not gpu_ids: console.print("[bold red]ERROR: No NVIDIA GPUs detected.[/bold red]"); sys.exit(1)
    console.print(f"[*] Found {len(gpu_ids)} NVIDIA GPUs: {gpu_ids}")

    ctx = mp.get_context("spawn")
    write_queue = ctx.Queue(maxsize=len(gpu_ids) * 20)
    status_queue = ctx.Queue()
    log_queue = ctx.Queue()

    # --- Worker Process Setup ---
    console.print(f"[*] Starting {args.cpu_workers} CPU writer processes...")
    cpu_writers = [ctx.Process(target=cpu_writer_process, args=(write_queue, log_queue, args.output)) for _ in range(args.cpu_workers)]
    for p in cpu_writers: p.start()

    num_gpus = len(gpu_ids)
    image_slices = [images_to_process[i::num_gpus] for i in range(num_gpus)]
    console.print(f"[*] Starting {num_gpus} GPU worker processes...")
    gpu_workers = [ctx.Process(target=gpu_worker_process, args=(gpu_ids[i], image_slices[i], write_queue, status_queue, args)) for i in range(num_gpus) if image_slices[i]]
    for p in gpu_workers: p.start()

    processing_start_time = time.time()

    # --- Live Display and Main Monitoring Loop ---
    progress = Progress(
        TextColumn("[bold blue]Processing..."), BarColumn(), "[progress.percentage]{task.percentage:>3.1f}%", "•",
        TextColumn("[cyan]{task.completed} of {task.total}"), "•",
        SpeedColumn(), "•", EstimatedTimeRemainingColumn(), console=console
    )
    progress_task = progress.add_task("New Images", total=len(images_to_process))
    gpu_stats = {gpu_id: {'completed': 0, 'start_time': time.time()} for gpu_id in gpu_ids}

    def generate_layout():
        gpu_table = Table.grid(expand=True)
        gpu_table.add_column("GPU ID", justify="right", style="cyan", no_wrap=True)
        gpu_table.add_column("Processed", justify="center", style="magenta")
        gpu_table.add_column("Speed (img/s)", justify="center", style="green")
        for gpu_id, stats in gpu_stats.items():
            elapsed = time.time() - stats['start_time']
            rate = stats['completed'] / elapsed if elapsed > 1 else 0.0
            # THE ONLY CHANGE IS IN THE LINE BELOW: Added " img/s"
            gpu_table.add_row(f"[bold]GPU {gpu_id}[/]", f"{stats['completed']:,} images", f"{rate:.2f} img/s")

        display_grid = Table.grid(padding=(0,0,1,0))
        display_grid.add_row(Panel(gpu_table, title="[bold]GPU Worker Status[/bold]", border_style="green"))
        display_grid.add_row(progress)
        return Panel(display_grid, title=f"[bold]Processing Session[/bold] ([yellow]{len(processed_files_set):,}[/] previously completed)", border_style="blue")

    log_file = open(log_file_path, "a", encoding="utf-8")

    all_workers = gpu_workers + cpu_writers
    total_completed = 0
    try:
        with Live(generate_layout(), console=console, screen=True, redirect_stderr=False, vertical_overflow="crop") as live:
            while total_completed < len(images_to_process):
                try:
                    while True: # Process all available status messages
                        msg = status_queue.get_nowait()
                        if msg["type"] == "gpu_progress":
                            gpu_stats[msg["gpu_id"]]['completed'] += 1
                            total_completed += 1
                        elif msg["type"] == "error": console.log(f"❌ [bold red]ERROR:[/bold red] {msg['message']}")
                        elif msg["type"] == "log": console.log(f"⚠️ [yellow]WARNING:[/] {msg['message']}")
                except queue.Empty:
                    pass

                try:
                    while True: # Process all available log messages
                        path_to_log = log_queue.get_nowait()
                        log_file.write(f"{path_to_log}\n")
                except queue.Empty:
                    pass

                log_file.flush()
                progress.update(progress_task, completed=total_completed)
                live.update(generate_layout())
                time.sleep(0.1)

    except KeyboardInterrupt:
        console.print("\n[!] Pausing session... Please wait for checkpointing to complete.", style="bold yellow")
    finally:
        log_file.close()

    # --- Clean Shutdown Sequence ---
    console.print("\n[*] Shutting down all workers...")

    for p in all_workers:
        if p.is_alive(): p.terminate()
    for p in all_workers:
        if p.is_alive(): p.join(timeout=5)

    duration_seconds = time.time() - processing_start_time
    minutes, seconds = divmod(duration_seconds, 60)

    final_processed_count = sum(stats['completed'] for stats in gpu_stats.values())

    console.print("\n[--- [bold yellow]Session Paused / Complete[/bold yellow] ---]")
    console.print(f"✅ Processed [bold]{final_processed_count:,}[/] new images this session in [bold]{int(minutes)} minutes and {seconds:.2f} seconds[/bold].")
    console.print(f"✅ Clean images are saved in: [link=file://{args.output.resolve()}]{args.output.resolve()}[/link]")

# ╰─────────────────────────────────────────────────────────────────────────╯

if __name__ == "__main__":
    main()
