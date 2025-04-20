import os
import subprocess
import mlflow

# Check that required environment variables are set
required_vars = ["MODEL", "EPOCHS", "BATCH", "OUTPUT"]
for var in required_vars:
    if var not in os.environ:
        raise EnvironmentError(f"Missing environment variable: {var}")

mlflow.start_run()

# Log training parameters
mlflow.log_param("model", os.environ["MODEL"])
mlflow.log_param("epochs", int(os.environ["EPOCHS"]))
mlflow.log_param("batch", int(os.environ["BATCH"]))

# Run YOLO training
cmd = f"""yolo segment train \
    data=/tmp/data.yaml \
    model={os.environ['MODEL']} \
    epochs={os.environ['EPOCHS']} \
    batch={os.environ['BATCH']} \
    exist_ok=True \
    device=cpu \
    project={os.environ['OUTPUT']} \
    name=experiment"""

subprocess.run(cmd, shell=True, check=True)

# Log the trained model as an MLflow artifact
artifact_path = f"{os.environ['OUTPUT']}/experiment/weights/best.pt"
if os.path.exists(artifact_path):
    mlflow.log_artifact(artifact_path)
else:
    print(f"Artifact not found at: {artifact_path}")

mlflow.end_run()
