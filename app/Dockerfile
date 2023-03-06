FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

# install utilities
RUN apt-get update && \
    apt-get install --no-install-recommends -y curl

RUN apt-get install python3.8 -y
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# Installing python dependencies
RUN python3 -m pip --no-cache-dir install --upgrade pip && \
    python3 --version && \
    pip3 --version

COPY ./requirements.txt .
RUN pip3 install -r requirements.txt

RUN pip install gunicorn
RUN pip install uvicorn
RUN pip install fastapi

# Copy app files
RUN mkdir /app
COPY . /app
WORKDIR /app/
ENV PYTHONPATH=/app
EXPOSE 80
RUN python3 -m gunicorn main:app -k uvicorn.workers.UvicornWorker