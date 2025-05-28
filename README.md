# YOLO11 Segmentation

A reproducible end-to-end pipeline for fine-tuning and deploying a YOLOv8 segmentation model, with both a local Docker workflow and full Azure ML integration.

## 🚀 Features

- **Pre-processing**: automatic train/val/test split and `data.yaml` generation  
- **Local testing**: run inference and quick fine-tuning inside a Docker container  
- **Azure ML integration**: programmatic provisioning of Resource Groups, Workspaces, Compute targets, Environments, Data assets, and remote training jobs  
- **Config-driven**: all names, paths, and credentials are read from a single `config.yaml`  
- **Modular code**: Jupyter notebooks for each step, plus a `src/train.py` for custom training  

## 📁 Repository Structure

```
├── docker/                         # Docker context for custom YOLO11 image
│   └── Dockerfile                  # Builds Ultralytics + dependencies
├── notebooks/                      # Step-by-step Jupyter notebooks
│   ├── 1_preprocessing.ipynb       # Data split & data.yaml
│   ├── 2_download_model.ipynb      # Fetch pre-trained weights
│   ├── 3_predict_local.ipynb       # Local inference in Docker
│   ├── 4_train_local.ipynb         # Quick local fine-tune in Docker
│   ├── 5_azure_resources.ipynb     # RG & Workspace provisioning
│   ├── 6_define_environment.ipynb  # Build & register custom Docker environment
│   ├── 7_create_compute.ipynb      # Create/reuse AML compute cluster
│   ├── 8_upload_data.ipynb         # Register training data asset
│   ├── 9_deploy_training.ipynb     # Submit remote training job
│   ├── 10_monitor_job.ipynb        # Stream logs from Azure ML
│   └── 11_download_artifacts.ipynb # Download trained weights & outputs
├── src/                            # Custom training script
│   └── train.py                    # Calls Ultralytics CLI or Python API
├── requirements.txt                # Python dependencies for local environment
├── config.yaml                     # User-specific settings (paths, Azure IDs, etc.)
├── README.md                       # ← you are here
└── LICENSE                         # MIT License
```

## 🛠 Prerequisites

- **Python 3.8+**
- **Docker** (Desktop on Windows/macOS; Engine on Linux)
- **Azure CLI** (for authentication via `az login`)
- An Azure subscription with **Contributor** rights

## ⚙️ Installation

1. Clone the repo and enter its directory:
   ```bash
   git clone https://github.com/Computer-Vision-WSC/yolo11-segmentation.git
   cd yolo11-segmentation
   ```
2. (Optional) Create & activate a Python virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Linux/macOS
   .venv\Scripts\activate.bat   # Windows
   ```
3. Install local dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🔧 Configuration

Create or edit `config.yaml` in the repo root with the following structure:

```yaml
azure:
  subscription_id:      "<YOUR_SUBSCRIPTION_ID>"
  resource_group_name:  "rg-yolo11-demo"
  workspace_name:       "mlw-yolo11-demo"
  location:             "westeurope"
  environment_name:     "yolo11-env"
  training_gpu_cluster: "gpu-cluster"
  compute_name:         "cpu-cluster"

train:
  data_asset_name: "yolo11-data"
  version:         "1"
  description:     "Raw images and annotations"
```

Adjust names, regions, and cluster sizes as needed.

## 📝 Usage

### 1. Local Docker workflow

Build the custom Docker image and run notebooks 1–4 to verify your environment and perform a quick smoke-test fine-tune:

```bash
docker build -t yolo11-docker:latest docker/
```

Open and execute:
- `1_preprocessing.ipynb`
- `2_download_model.ipynb`
- `3_predict_local.ipynb`
- `4_train_local.ipynb`

### 2. Azure ML pipeline

Run notebooks 5–11 in order. They will:
1. Provision Azure resources (Resource Group, Workspace, Compute, Environment)
2. Register your training data as an Azure ML Data asset
3. Submit and monitor a remote training job
4. Download the trained weights and logs locally

Each notebook is fully commented and designed for copy-paste-and-run.

## 🤝 Contributing

Contributions and bug reports are welcome! Please open an issue or pull request on GitHub.

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
