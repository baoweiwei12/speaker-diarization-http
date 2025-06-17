# 使用NVIDIA CUDA基础镜像
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu22.04
 
# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# 创建并激活虚拟环境
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 升级pip
RUN pip install --no-cache-dir --upgrade pip

# 安装Python依赖
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    pydantic \
    pyannote.audio \
    redis \
    python-dotenv

# 只复制应用代码
COPY ./main.py /app/main.py

# 暴露端口
EXPOSE 9001

# 启动命令
CMD ["python3", "main.py"] 