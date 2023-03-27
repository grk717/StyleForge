from models.dummy import DummyModel
from torchvision.models import efficientnet_v2_s
from torchvision.models import efficientnet_b0
import torch.nn as nn

def get_model(name: str):
    if name == 'dummy':
        return DummyModel()
    elif name == 'enet':
        mdl = efficientnet_b0()
        mdl.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features=1280, out_features=512)
        )
        return mdl