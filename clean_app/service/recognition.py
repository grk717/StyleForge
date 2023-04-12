""" Build a recognition service from infrastructure models"""

from dataclasses import asdict
from typing import List
import logging

import numpy as np

from service.interface import ServicePredictionDataModel, RecognitionServiceInterface


class RecognitionService(RecognitionServiceInterface):

    def recognize_image(self, image: np.ndarray) -> List[ServicePredictionDataModel]:
        """Function to inference image on detector and searcher
        Args:
            image (np.ndarray): [description]
        Returns:
            List[ServicePrediction]: [description]
        """
        logging.info("Start recognizing image ... ")
        results = []

        detected_boxes = self.detector.detect(image=image)
        logging.info("Start search")
        for detected_box in detected_boxes:

            x_min, y_min, x_max, y_max = detected_box.predicted_box
            cropped_image = image[y_min:y_max, x_min:x_max]

            searcher_predictions = self.searcher.search(cropped_image)

            pred = ServicePredictionDataModel(
                bbox=detected_box.predicted_box,
                search_results=searcher_predictions
            )
            results.append(pred)
        return results


if __name__ == "__main__":
    from infrastructure.detection.model import YoloClothesDetector
    from infrastructure.search.model import ResnetSearcher
    from PIL import Image
    
    # Init dummy models
    detector = YoloClothesDetector("../app/detector.pt", 0.5)
    searcher = ResnetSearcher("../database/epoch99.pth", "../embeddings.npy", "../df.csv")

    # Init recognition service
    recognition_service = RecognitionService(detector, searcher)

    # create random image
    imarray = np.array(Image.open("../database/database_images/19783188_model.jpg"))

    # get dummy predictions
    predictions = recognition_service.recognize_image(image=imarray)
    for i, pred in enumerate(predictions):
        print(f"Predictions {i}: \n {asdict(pred)} \n")