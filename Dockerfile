FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

RUN apt update \
    && apt install unzip \
    && apt install libgl1 \

    #TODO: add working directory, copy data

    RUN pip install -r requirements.txt \
    && pip install git+https://github.com/facebookresearch/segment-anything.git \
    && pip install git+https://github.com/openai/CLIP.git

# Missing: setup of data dirs
# export PYTHONPATH="$PWD"