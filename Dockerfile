FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN dpkg --remove-architecture i386 || true && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg \
        lsb-release \
	python3 python3-pip python3-dev git && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    pip3 install tensorflow==2.15 && \
    pip3 install matplotlib scikit-learn pandas numpy tqdm

# (Opcjonalnie) Jupyter Notebook
RUN pip3 install jupyter notebook

WORKDIR /workspace
CMD ["nvidia-smi"]
#CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
