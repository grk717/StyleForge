from pydantic import BaseSettings, Field
from typing import Literal


class TrainConfig(BaseSettings):

    MODEL: str = Field(default="unet")
    ENCODER: str = Field(default='efficientnet-b0')
    ENCODER_WEIGHTS: str = Field('imagenet')
    ACTIVATION: Literal[None, "sigmoid", "softmax2d"] = Field(default="softmax2d")
    SEED: int = Field(default=123)
    DEVICE: str = Field(default="cuda")
    TRAIN_CLASSES_FRAC: float = Field(default=0.7)
    # classes
    CLASSES: dict = Field(default={ 
            0: "background",
            1: "short sleeve top", 
            2: "long sleeve top", 
            3: "short sleeve outwear",
            4: "long sleeve outwear",
            5: "vest",
            6: "sling",
            7 : "shorts",
            8 : "trousers",
            9 : "skirt", 
            10 : "short sleeve dress",
            11 : "long sleeve dress", 
            12 : "vest dress",
            13 : "sling dress"
            })