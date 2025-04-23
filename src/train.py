import os
import argparse
import subprocess
import mlflow

os.environ["MLFLOW_TRACKING_URI"] = "noop"

if mlflow.active_run():
    mlflow.end_run()

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--batch", type=int, required=True)
parser.add_argument("--output_dir", type=str, required=True)

args = parser.parse_args()

mlflow.start_run()

# Log training parameters
mlflow.log_param("model", args.model)
mlflow.log_param("epochs", args.epochs)
mlflow.log_param("batch", args.batch)

# Paths
experiment_name = "experiment"
experiment_path = os.path.join(args.output_dir, experiment_name)

# Run YOLO training
cmd = [
    "yolo", "segment", "train",
    "data=/tmp/data.yaml",
    f"model={args.model}",
    f"epochs={args.epochs}",
    f"batch={args.batch}",
    "exist_ok=True",
    "device=cpu",
    f"project={args.output_dir}",
    f"name={experiment_name}"
]

print("Running:", " ".join(cmd))
subprocess.run(" ".join(cmd), shell=True, check=True)

# Log the trained model
model_path = os.path.join(experiment_path, "weights", "best.pt")
if os.path.exists(model_path):
    mlflow.log_artifact(model_path)
    print("✅ Model artifact logged.")
else:
    print(f"⚠️ Model not found at: {model_path}")

mlflow.end_run()




# import os
# import subprocess
# import mlflow

# # Check that required environment variables are set
# required_vars = ["MODEL", "EPOCHS", "BATCH", "OUTPUT"]
# for var in required_vars:
#     if var not in os.environ:
#         raise EnvironmentError(f"Missing environment variable: {var}")

# mlflow.start_run()

# # Log training parameters
# mlflow.log_param("model", os.environ["MODEL"])
# mlflow.log_param("epochs", int(os.environ["EPOCHS"]))
# mlflow.log_param("batch", int(os.environ["BATCH"]))

# # Run YOLO training
# cmd = f"""yolo segment train \
#     data=/tmp/data.yaml \
#     model={os.environ['MODEL']} \
#     epochs={os.environ['EPOCHS']} \
#     batch={os.environ['BATCH']} \
#     exist_ok=True \
#     device=cpu \
#     project={os.environ['OUTPUT']} \
#     name=experiment"""

# subprocess.run(cmd, shell=True, check=True)

# # Log the trained model as an MLflow artifact
# artifact_path = f"{os.environ['OUTPUT']}/experiment/weights/best.pt"
# if os.path.exists(artifact_path):
#     mlflow.log_artifact(artifact_path)
# else:
#     print(f"Artifact not found at: {artifact_path}")

# mlflow.end_run()
