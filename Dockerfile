FROM pytorch/pytorch:latest

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y python3-opencv && apt-get install -y git
RUN pip install opencv-python

RUN pip install --upgrade pip
RUN pip3 install numpy
RUN pip3 install torch
RUN pip3 install pillow
RUN pip3 install future
RUN pip3 install torchvision==0.15.1

RUN apt install -y gcc

RUN pip3 install pycocotools
#RUN pip install -y pycocotools==2.0.6
RUN pip3 install albumentations
RUN pip3 install pandas


ENV PYTHONUNBUFFERED=1

CMD ["python3", "train.py"]

