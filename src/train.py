import os
import argparse
import subprocess

def main():

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to YOLO model", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--output_dir", type=str, help="Directory to save outputs", required=True)
    args = parser.parse_args()

    # Define experiment subfolder
    experiment_name = "experiment"
    experiment_path = os.path.join(args.output_dir, experiment_name)

    # Run YOLO training command
    cmd = [
        "yolo", "segment", "train",
        f"data=/tmp/data.yaml",
        f"model={args.model}",
        f"epochs={args.epochs}",
        f"batch={args.batch}",
        "exist_ok=True",
        "device=cpu",
        f"project={args.output_dir}",
        f"name={experiment_name}"
    ]

    print("Running command:", " ".join(cmd))
    subprocess.run(" ".join(cmd), shell=True, check=True)

if __name__ == "__main__":
    main()