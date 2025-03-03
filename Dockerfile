FROM tensorflow/tensorflow:2.15.0-gpu-jupyter 

WORKDIR /app

# Ython version
ARG PYTHON_VERSION=3.11

# OPENCV dependencies
RUN apt update && apt install -y libgl1 libglib2.0-0
RUN apt update && apt install -y python3-tk
RUN pip install numpy==1.26.4 opencv-python==4.10.0.84 pandas==2.2 scikit-learn==1.6.1 streamlit==1.42.2
RUN pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118

RUN python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
RUN python -c "import torch; print('PyTorch:', torch.__version__); print('GPU disponible:', torch.cuda.is_available())"

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]