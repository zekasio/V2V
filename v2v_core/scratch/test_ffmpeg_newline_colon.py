import subprocess

text = "Line 1: \nLine 2"
# Escape the colon
escaped_text = text.replace(":", "\\:")
# The newline should remain as \n
# Attempt to run FFmpeg with this text
# Use shell=True to simulate the command string construction
cmd = f"ffmpeg -f lavfi -i color=c=black:s=640x480:d=1 -vf \"drawtext=text='{escaped_text}':x=10:y=10:fontsize=24:fontcolor=white\" -f null -"
print(f"Running: {cmd}")
try:
    subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    print("Success!")
except subprocess.CalledProcessError as e:
    print("Failed!")
    print(e.stderr)
