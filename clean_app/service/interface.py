""" Build a recognition service from infrastructure models"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np

from infrastructure.detection.model import Detector
from infrastructure.search.model import Searcher, SearchPrediction


@dataclass
class ServicePredictionDataModel:
    bbox: List[int]
    search_results: List[SearchPrediction]


class RecognitionServiceInterface:
    def __init__(self, detector: Detector, searcher: Searcher):
        self.detector = detector
        self.searcher = searcher

    @abstractmethod
    def recognize_image(self, image: np.ndarray) -> List[ServicePredictionDataModel]:
        """Function to inference image on detector and classifier
        Args:
            image (np.ndarray): [description]
        Returns:
            List[ServicePrediction]: [description]
        """
        results = []
        return results
