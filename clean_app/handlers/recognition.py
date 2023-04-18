from abc import ABC, abstractmethod
from typing import Tuple, List
import logging

from dataclasses import asdict
import numpy as np

from pydantic import ValidationError

from service.recognition import RecognitionServiceInterface
from service.image_loader import ImageLoader
from handlers.data_models import RecognitionResultsModel, RecognitionRequestModel


class RecognitionHandler(ABC):
    def __init__(self, recognition_service: RecognitionServiceInterface, image_loader: ImageLoader):
        self.recognition_service = recognition_service
        self.image_loader = image_loader
        self.response_scheme = RecognitionRequestModel
        self.result_scheme = RecognitionResultsModel

    @abstractmethod
    def handle(self, body: str):
        pass

    @abstractmethod
    def _load_image(self, url: str) -> np.ndarray:
        pass

    def deserialize_body(self, body: str) -> Tuple[RecognitionRequestModel, str]:
        logging.info("Deserializing body...")

        body_schema = None
        errs = None
        try:
            body_schema = self.response_scheme.load(body)
        except ValidationError as err:
            errs = err.messages
        return body_schema, errs

    def serialize_answer(self, result, task_id) -> str:
        logging.info("Serializing body...")
        return self.result_scheme.dump({"task_id": task_id, "recognitions": result})


class StyleForgeRecognitionHandler(RecognitionHandler):
    def __init__(self, recognition_service: RecognitionServiceInterface, image_loader: ImageLoader):
        super().__init__(recognition_service, image_loader)

    def handle(self, body: RecognitionRequestModel) -> List:
        """
        Takes response body from Queue
        - deserialize and validate body fields
        - make service calls, Load image and extract products
        - serialize results
        - return results to transport layer
        :param body:
        :return:
        """
        results = []
        if body:
            task_id = body.task_id
            # call image loader
            image = self._load_image(body.image_url)
            # consume products from shelf
            rec_results = self.recognition_service.recognize_image(image=image)
            # append meta fields to products
            results = RecognitionResultsModel(task_id=task_id, recognitions=[asdict(i) for i in rec_results])
        return results

    def _load_image(self, url: str) -> np.ndarray:
        """
        Call an image loader
        :param url:
        :return:
        """
        image_array = None
        try:
            img_bytes = self.image_loader.load_image(url=url)
            image_array = self.image_loader.image_as_array(img_bytes)
            logging.info("Image succefully loaded")
        except Exception as err:
            print(err)
        return image_array
