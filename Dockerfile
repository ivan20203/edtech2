
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system deps
RUN apt-get update && apt-get install -y \
    wget git curl unzip ffmpeg libsndfile1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install full CUDA toolkit and compiler (nvcc etc.)
RUN apt-get update && apt-get install -y wget gnupg && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin && \
    mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.1-530.30.02-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.1-530.30.02-1_amd64.deb && \
    cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get install -y cuda-toolkit-12-1 cuda-compiler-12-1 && \
    rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install Miniconda manually
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh && \
    conda config --set always_yes yes && \
    conda config --set report_errors false && \
    conda config --add channels https://repo.anaconda.com/pkgs/main && \
    conda config --add channels https://repo.anaconda.com/pkgs/r && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda update -n base -c defaults conda

# Create working directory
WORKDIR /workspace

# Copy entire repo
COPY . /workspace

# Unpack conda environment
COPY mooncast.tar.gz /tmp/mooncast.tar.gz
RUN mkdir -p $CONDA_DIR/envs/mooncast && \
    tar -xzf /tmp/mooncast.tar.gz -C $CONDA_DIR/envs/mooncast

# Activate env + fix symlinks
RUN echo "source activate mooncast" >> ~/.bashrc
ENV PATH=$CONDA_DIR/envs/mooncast/bin:$PATH

# Set env variables if needed (like OpenAI key)
ENV TRANSFORMERS_CACHE=/workspace/models
ENV HF_HOME=/workspace/models

# If you're serving via FastAPI or Flask
EXPOSE 8000

# Default command (change this if you want to serve API instead)
CMD ["conda", "run", "-n", "mooncast", "python", "MoonDIA/trained_mapper/MoonCast_seed.py"]
