# Use the Miniconda base image
FROM continuumio/miniconda3:latest

# Disable interactive prompts in apt/conda
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install system libraries required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 2. Create the conda environment 'yolo11env' with Python 3.12, install pip, ultralytics, and onnx
RUN conda create --name yolo11env -y python=3.12 \
    && conda run -n yolo11env conda install pip -y \
    && conda run -n yolo11env pip install ultralytics onnx mlflow

# Update PATH so Python 3.12 is used by default
ENV PATH /opt/conda/envs/yolo11env/bin:$PATH

# Switch the shell to bash -c and add conda activation to ~/.bashrc
SHELL ["/bin/bash", "-c"]

# Add the 'conda activate yolo11env' command to root's ~/.bashrc
RUN echo "conda activate yolo11env" >> /root/.bashrc

# Set /bin/bash as the default command; it reads ~/.bashrc and activates the environment
CMD ["/bin/bash"]
