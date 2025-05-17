FROM tensorflow/tensorflow:2.15.0-gpu-jupyter 

WORKDIR /app

# OPENCV, tkinter and others dependencies
RUN apt update && apt install -y \
    libgl1 libglib2.0-0 \
    python3-tk \
    cmake g++ make libopenblas-dev liblapack-dev libx11-dev && \
    rm -rf /var/lib/apt/lists/*

# install important libraries
RUN pip install numpy==1.26.4 \
    opencv-python==4.10.0.84 \
    pandas==2.2 \
    scikit-learn==1.6.1 \
    streamlit==1.42.2 \
    dlib==19.24.6 \
    tk

# install pytorch 
RUN pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118

# check the tf and py versions
RUN python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
RUN python -c "import torch; print('PyTorch:', torch.__version__); print('GPU disponible:', torch.cuda.is_available())"

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]