import subprocess

def escape_ffmpeg_text(text: str) -> str:
    # Escape \, :, and ,
    return text.replace('\\', '\\\\').replace(':', '\\:').replace(',', '\\,')

text = "Line 1: \nLine 2, and \\ backslash"
escaped_text = escape_ffmpeg_text(text)
print(f"Escaped text: {escaped_text}")

# Use shell=True to simulate the command string construction
# Note: I need to use f"..." and be careful about single quotes in the command.
# If I use double quotes around the filter_complex value, I need to escape any double quotes inside.
# But for now, let's keep it simple.
cmd = f"ffmpeg -f lavfi -i color=c=black:s=640x480:d=1 -vf \"drawtext=text='{escaped_text}':x=10:y=10:fontsize=24:fontcolor=white\" -f null -"
print(f"Running: {cmd}")
try:
    subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    print("Success!")
except subprocess.CalledProcessError as e:
    print("Failed!")
    print(e.stderr)
