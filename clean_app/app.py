from infrastructure.detection.model import YoloClothesDetector
from infrastructure.search.model import ResnetSearcher, ViTSearcher

from service.image_loader import RequestImageLoader
from service.recognition import RecognitionService

from handlers.data_models import RecognitionRequestModel

from handlers.recognition import StyleForgeRecognitionHandler
from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from config import AppConfig

config = AppConfig()

detection_model = YoloClothesDetector(config.DETECTOR_MODEL_PATH, 0.5)
search_model = ResnetSearcher(config.EMBEDDER_MODEL_PATH, config.EMBEDDING_DATABASE_PATH,\
                              config.DATABASE_PATH, top_k=5)

recognition_service = RecognitionService(detection_model, search_model)
image_loader = RequestImageLoader()

handler = StyleForgeRecognitionHandler(recognition_service, image_loader)

app = FastAPI(title="StyleForge search service")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
    )


@app.get('/healthcheck', status_code=status.HTTP_200_OK)
def healthcheck():
    return {'healthcheck': 'Everything OK!'}


@app.post("/process-image")
async def process_image(item: RecognitionRequestModel):
    return handler.handle(item)

# docker build -t styleforge .
# docker run --name styleforge_test --gpus all --publish 80:80 styleforge
