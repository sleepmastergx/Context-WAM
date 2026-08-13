# OPTIONAL — for future RunPod deployment. The repository runs WITHOUT docker
# (git clone + bash setup.sh, see README); this image just bakes the same
# dependency steps. Not yet pushed to a registry or smoke-tested as an image.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

WORKDIR /workspace/Context-WAM
ENV PYTHONNOUSERSITE=1
COPY requirements.txt .
RUN pip install -U pip wheel setuptools && \
    pip install "deepspeed==0.18.5" --no-build-isolation && \
    pip install -r requirements.txt
COPY . .
RUN pip install -e third_party/fastwam --no-deps

# Data / weights / outputs live on the mounted volume:
#   docker run --gpus all -v /workspace:/workspace <image> \
#     bash -c "SKIP_VENV=1 SKIP_DATA=1 bash setup.sh && source env.sh && \
#              python train.py --arm ttt"
CMD ["bash"]
