## FROM python:3.9.19-bullseye
## KAGGLE ENVIRONMENT:
FROM python:3.10.14-bullseye

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Download inception v3 imagenet weights
## ADD https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels.h5 .
ADD https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 .

## Download finetuned model
ADD https://storage.googleapis.com/figures-gp/physionet/multilabel-model-v23.keras ./model/multilabel-model.keras

## download ptbxl-database file
ADD https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv .

## Install your dependencies here using apt install, etc.
# RUN pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cpu

## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
