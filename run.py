import os
import sys
import asyncio
from pathlib import Path

# Configure console encoding for UTF-8 support on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add v2v_core directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.resolve() / "v2v_core"))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.resolve() / "v2v_core" / ".env")
except ImportError:
    pass

from src.logger import get_logger
from src.pipeline import LocalizationPipeline

logger = get_logger("run_cli")

def print_banner():
    print("┌──────────────────────────────────────────────────────────────┐")
    print("│                     V2V Localization CLI                     │")
    print("│         Select a video to translate and dub it               │")
    print("└──────────────────────────────────────────────────────────────┘")

def find_mp4_files():
    # Scan root directory for mp4 files
    root_dir = Path(__file__).parent.resolve()
    mp4_files = list(root_dir.glob("*.mp4"))
    # Also check 'input' directory if it exists inside v2v_core
    input_dir = root_dir / "v2v_core" / "input"
    if input_dir.exists():
        mp4_files.extend(input_dir.glob("*.mp4"))
    
    # Exclude temp processed or output files (files ending with _processed.mp4, _EN.mp4, or _dubbed.mp4)
    filtered = []
    for f in mp4_files:
        name = f.name.lower()
        if "_processed" in name or "_en" in name or "_dubbed" in name:
            continue
        filtered.append(f)
    
    # Remove duplicates and sort
    unique_files = sorted(list(set(filtered)), key=lambda x: x.name)
    return unique_files

async def main():
    print_banner()
    
    videos = find_mp4_files()
    if not videos:
        print("[-] No MP4 video files found in the current directory or 'input' folder.")
        print("Please place your video files in the root folder and try again.")
        sys.exit(1)
        
    print("\n[+] Available Videos:")
    for idx, video in enumerate(videos, 1):
        # Show path relative to root directory for cleanliness
        relative_path = video.name
        if video.parent.name == "input":
            relative_path = f"v2v_core/input/{video.name}"
        print(f"  [{idx}] {relative_path} ({os.path.getsize(video) / (1024*1024):.1f} MB)")
        
    print("\n[?] Enter the number of the video you want to process (or 'q' to quit): ", end="")
    try:
        user_input = input().strip()
        if user_input.lower() == 'q':
            print("Exiting.")
            sys.exit(0)
            
        choice = int(user_input)
        if choice < 1 or choice > len(videos):
            raise ValueError()
    except (ValueError, IndexError):
        print("[-] Invalid selection. Exiting.")
        sys.exit(1)
        
    selected_video = videos[choice - 1]
    print(f"\n[+] Selected: {selected_video.name}")
    print("[+] Starting pipeline... Please wait.\n")

    pipeline = LocalizationPipeline()
    try:
        output_path = await pipeline.process_video(selected_video)
        print("\n" + "="*60)
        print(f"[+] SUCCESS: Translation and dubbing complete!")
        print(f"[+] Output saved to: {output_path}")
        print("="*60 + "\n")
    except Exception as e:
        print(f"\n[-] ERROR: Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[-] Cancelled by user.")
        sys.exit(0)
