# INVC YOLO11 垃圾分类项目 - 训练镜像
#
# 技术决策(见《环境准备任务书》第 3 节):
#  - 基础镜像固定为 pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
#  - 不要使用 ultralytics/ultralytics:latest(其为 CUDA 13.2,与本机 577 驱动不兼容)
#  - 不安装 Windows 主机 CUDA Toolkit,容器自带 CUDA 运行库
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# 国内 pip 镜像(清华源),可按需换成阿里云/中科大等
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /workspace

# OpenCV 所需 Linux 运行库(任务书第 6 节)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖清单,便于利用构建缓存
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -i ${PIP_INDEX_URL} -r /workspace/requirements.txt

# 项目代码/数据/权重在运行时通过 compose 挂载;
# 构建时不写入数据集、私人凭据或模型运行结果(任务书第 6 节)
CMD ["/bin/bash"]
