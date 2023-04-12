from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass
import logging
import numpy as np
import torch


@dataclass
class DetectorPrediction:
    predicted_box: List[int]
    box_probability: float


class Detector(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[DetectorPrediction]:
        """
        Return boxes of all detected contours from image.
        :param image: np.ndarray RGB image
        :return: List[Contour]
        """
        pass


class YoloClothesDetector(Detector):

    def __init__(self, detection_model_path: str, threshold: float=0.5):
        logging.info('Loading Detector')
        self.model_path = detection_model_path
        self.threshold = threshold
        self.detection_model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path, force_reload=True)
        self.detection_model.conf = self.threshold


    def detect(self, image: np.ndarray) -> List[DetectorPrediction]:
        results = self.detection_model(image).xyxy[0].cpu().numpy()
        return [DetectorPrediction(predicted_box=i[:4].astype(int).tolist(), box_probability=i[-2]) for i in results]