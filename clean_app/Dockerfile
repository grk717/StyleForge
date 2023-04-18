FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# install utilities
RUN apt-get update && \
    apt-get install --no-install-recommends -y curl

RUN apt-get install python3.8 -y
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# Installing python dependencies
COPY ./requirements.txt .
RUN pip3 install -r requirements.txt

# Copy app files
RUN mkdir /app
COPY . /app
WORKDIR /app/
ENV PYTHONPATH=/app
EXPOSE 80
#ENTRYPOINT ["python3", "-m", "gunicorn", "main:app"]
ENTRYPOINT ["uvicorn", "app:app", "--host=0.0.0.0", "--port=80"]