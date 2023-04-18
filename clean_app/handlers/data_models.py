from pydantic import BaseModel, AnyUrl
from typing import List
from infrastructure.search.model import SearchPrediction


class RecognitionsModel(BaseModel):
    bbox: List[int]
    search_results: List[SearchPrediction]


class RecognitionResultsModel(BaseModel):
    task_id: str
    recognitions: List[RecognitionsModel]


class RecognitionRequestModel(BaseModel):
    task_id: str
    image_url: AnyUrl
