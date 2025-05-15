import os
import subprocess

# Set your paths
data_dir = "/home/hastabbe/2024-zeyer-ctc-librispeech-spm10k"
input_dir = "/home/hastabbe/sound_mnist_16khz"
output_dir = "/home/hastabbe/encoded"
model_script = "model_2024_ctc_spm10k.py"  # Make sure this is in the same dir or give full path

# Create output dir if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through all wav files
for filename in os.listdir(input_dir):
    if filename.endswith(".wav"):
        input_path = os.path.join(input_dir, filename)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{base_name}_enc.csv")

        # Build the command
        cmd = [
            "python", model_script,
            input_path,
            "--data-dir", data_dir,
            "--save-encoded", output_path
        ]

        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)
