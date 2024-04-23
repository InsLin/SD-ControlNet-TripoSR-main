FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

ARG FASTCHAT_VERSION="main"

RUN pip3 install --upgrade pip




COPY SD-ControlNet-TripoSR-main /SD-ControlNet-TripoSR-main

WORKDIR /SD-ControlNet-TripoSR-main

# 安装所需的依赖
RUN pip3 install .


# 在启动容器时运行gradio.py
#CMD ["python", "gradio.py"]
