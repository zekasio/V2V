import subprocess

text = "Line 1\nLine 2"
# Attempt to run FFmpeg with this text
# This is a simplified command. I just want to see if it fails.
cmd = [
    'ffmpeg',
    '-f', 'lavfi', '-i', 'color=c=black:s=640x480:d=1',
    '-vf', f"drawtext=text='{text}':x=10:y=10:fontsize=24:fontcolor=white",
    '-f', 'null', '-'
]
print(f"Running: {' '.join(cmd)}")
try:
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    print("Success!")
except subprocess.CalledProcessError as e:
    print("Failed!")
    print(e.stderr)
