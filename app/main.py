from fastapi import FastAPI, UploadFile
from PIL import Image
import io
import torch
from model.processing_functions import detection, embedding_creation


detection_model = torch.hub.load("ultralytics/yolov5", "yolov5n")
detection_model.conf = 0.5

embedding_model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
embedding_model.eval().to('cuda')


app = FastAPI(title="StyleForge search service")


@app.on_event("startup")
async def init_models():
    """
    Initialize detection and embedding extraction models
    :return:
    """


@app.post("/process-image")
async def process_image(file: UploadFile):
    """
    Image processing pipeline
    Detection -> Embedding Extraction -> Nearest Neighbors Search

    :param file:
    :return:
    """

    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    detection_result = detection(image, detection_model)
    embeddings = embedding_creation(detection_result, embedding_model)
    return {"embedding": embeddings}



