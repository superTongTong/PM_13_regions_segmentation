FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update --fix-missing -y

#RUN apt-get install -y --allow-change-held-packages  libcudnn8=8.1.1.33-1+cuda11.2 libcudnn8-dev=8.1.1.33-1+cuda11.2

RUN apt-get install -y ffmpeg libsm6 libxrender1 libxtst6 zip

# Library components for av
RUN apt-get install -y \
    libavformat-dev libavcodec-dev libavdevice-dev \
    libavutil-dev libswscale-dev libswresample-dev libavfilter-dev

RUN apt-get install -y python3 python3-pip git python3-dev pkg-config htop

RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install monai numpy nibabel wandb timm pytz python-dateutil SimpleITK pandas tqdm p_tqdm
RUN pip3 install psutil scipy joblib batchgenerators argparse tifffile scikit-image matplotlib monai[einops]
RUN pip3 install requests slicerio dicom2nifti seaborn scikit-learn dynamic_network_architectures acvl_utils pyarrow


WORKDIR /app/script