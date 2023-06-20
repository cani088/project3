FROM python:3.7.9-slim-buster

RUN apt-get update -y && apt-get install -y --no-install-recommends curl python3-pip python3-dev libsm6 libxext6 libxrender-dev libatlas-base-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libopencv-dev build-essential pkg-config libjpeg-dev libpng-dev libgtk-3-dev && rm -rf /var/lib/apt/lists/*

COPY . /app

COPY ./requirements.txt /app/requirements.txt
WORKDIR /app

RUN pip3 install --upgrade pip; pip3 install --no-cache-dir -r requirements.txt;
RUN ls

ENTRYPOINT [ "python3" ]

CMD [ "app_working_Vladimir.py" ]