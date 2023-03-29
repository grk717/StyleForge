from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import torch
from model.processing_functions import detection, embedding_creation


detection_model = torch.hub.load('ultralytics/yolov5', 'custom', path='detector.pt', force_reload=True)

detection_model.conf = 0.5

#embedding_model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
#embedding_model.eval().to('cuda')

print("Initalized models, start serving")

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
    #embeddings = embedding_creation(detection_result, embedding_model)
    crops = []
    for i in detection_result:
        bytes_io = io.BytesIO()
        img_base64 = Image.fromarray(i)
        img_base64.save(bytes_io, format="jpeg")
        crops.append(bytes_io.getvalue())
    print(len(crops))
    return StreamingResponse((i for i in crops[::-1]), media_type="image/jpeg")
    #return Response(content=bytes_io.getvalue(), media_type="image/jpeg")
    #return {"embedding": detection_result}


# docker build -t styleforge .
# docker run --name styleforge_test --gpus all --publish 80:80 styleforge
