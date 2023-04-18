from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass
import logging
import numpy as np
import torch
from torchvision import transforms
import pandas as pd
import faiss
from PIL import Image


@dataclass
class SearchPrediction:
    predicted_image_link: str
    predicted_link: str


class Searcher(ABC):

    @abstractmethod
    def __init__(self, search_model_path: str, database_path: str, top_k: int):
        pass

    @abstractmethod
    def search(self, image: np.ndarray) -> List[SearchPrediction]:
        """
        Return 
        :param image: np.ndarray RGB image
        :return: List[Contour]
        """
        pass


class ResnetSearcher(Searcher):

    def __init__(self, embedder_model_path: str, emb_database_path: str, database_path: str, top_k: int=3):
        self.embedder_model_path = embedder_model_path
        self.top_k = top_k
        self.embedder_model = torch.load(self.embedder_model_path)
        
        self.database = pd.read_csv(database_path, index_col="id")
        self.preprocessing = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # init FAISS index
        # TODO move to Searcher class, change current class to embedder class
        emb_database = np.load(emb_database_path)
        self.index = faiss.IndexFlatIP(512)
        self.index.add(emb_database)


    def search(self, image: np.ndarray) -> List[SearchPrediction]:
        image = self.preprocessing(Image.fromarray(image)).cuda()
        with torch.no_grad():
            embedding = self.embedder_model(image.cuda().unsqueeze(0)).cpu().numpy()
        D, I = self.index.search(embedding / np.linalg.norm(embedding), self.top_k)
        return [SearchPrediction(i[0], i[1]) for i in self.database.iloc[I[0]].values]


class ViTSearcher(Searcher):
    def __init__(self, embedder_model_path: str, emb_database_path: str, database_path: str, top_k: int = 3):
        self.embedder_model_path = embedder_model_path
        self.top_k = top_k
        self.embedder_model = torch.load(self.embedder_model_path)

        self.database = pd.read_csv(database_path, index_col="id")
        self.preprocessing = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # init FAISS index
        # TODO move to Searcher class, change current class to embedder class
        emb_database = np.load(emb_database_path)
        self.index = faiss.IndexFlatIP(512)
        self.index.add(emb_database)

    def search(self, image: np.ndarray) -> List[SearchPrediction]:
        image = self.preprocessing(Image.fromarray(image)).cuda()
        with torch.no_grad():
            embedding = self.embedder_model(image.cuda().unsqueeze(0)).cpu().numpy()
        D, I = self.index.search(embedding / np.linalg.norm(embedding), self.top_k)
        return [SearchPrediction(i[0], i[1]) for i in self.database.iloc[I[0]].values]