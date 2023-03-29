import argparse
from clearml import Task, Logger
import json
import math
import matplotlib.pyplot as plt
from models.dummy import DummyModel
from models import get_model
import numpy as np
import os
import pandas as pd
from PIL import Image
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import seaborn as sns
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import crop
from torchvision.io import read_image
from torchvision import datasets, transforms
from tqdm import tqdm
from utils.data import DeepFashionDataset



seed = 123
train_classes_frac = 0.7
num_epochs = 100
batch_size = 256
base_lr = 0.01
device = torch.device("cuda")


### training cycle ###
def train(model, loss_func, device, train_loader, optimizer, loss_optimizer, epoch, iterations_in_epoch, logger):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        loss_optimizer.zero_grad()
        embeddings = model(data)
        loss = loss_func(embeddings, labels)
        loss.backward()
        optimizer.step()
        loss_optimizer.step()
        
        if batch_idx % 50 == 0:
            logger.report_scalar(title="Loss", series="Train loss", iteration=iterations_in_epoch*(epoch) + batch_idx, value=loss.item())
            print("Epoch {} Iteration {}: Loss = {}".format(epoch, batch_idx, loss))

### get all embeddings from dataset ###
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester(dataloader_num_workers=8, batch_size=batch_size)
    return tester.get_all_embeddings(dataset, model)


### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
def test(query_dataset, retrieval_dataset, model, accuracy_calculator, epoch, iterations_in_epoch, logger):
    query_embeddings, query_labels = get_all_embeddings(query_dataset, model)
    retrieval_embeddings, retrieval_labels = get_all_embeddings(retrieval_dataset, model)

    query_labels = query_labels.squeeze(1)
    retrieval_labels = retrieval_labels.squeeze(1)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        query_embeddings, query_labels, retrieval_embeddings, retrieval_labels, False
    )
    logger.report_scalar(title="Metrics", series="Precision@1", iteration=iterations_in_epoch*(epoch+1), value=accuracies["precision_at_1"])
    logger.report_scalar(title="Metrics", series="mAP", iteration=iterations_in_epoch*(epoch+1), value=accuracies["mean_average_precision"])
    print("Creating visualisation")
    if epoch % 5 == 0:
        comb_emb = np.concatenate((query_embeddings.cpu().numpy(), retrieval_embeddings.cpu().numpy()))
        comb_lab = np.concatenate((query_labels.cpu().numpy(), retrieval_labels.cpu().numpy()))
        comb_src = np.concatenate(
            (np.repeat("Query", len(query_embeddings)), np.repeat("Retrieval", len(retrieval_embeddings)))
        )

        comb_tsne = TSNE(metric='cosine', random_state=seed).fit_transform(comb_emb)

        sns.scatterplot(
            x=comb_tsne[:, 0],
            y=comb_tsne[:, 1],
            hue=comb_lab,
            style=comb_src,
            palette="Paired",
        )
        plt.title("Query & Retrieval Embeddings tSNE")
        plt.legend([],[])
        logger.report_matplotlib_figure("Class Embeddings tSNE", "series", plt.gcf(), iteration=iterations_in_epoch*(epoch+1))
        plt.clf()
    
    print("Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))
    print("Test set map = {}".format(accuracies["mean_average_precision"]))




def main(opt):
    # Logging
    task = Task.init(project_name='styleforge/metric_learning', task_name=opt.task_name)
    configuration_dict = {'number_of_epochs': num_epochs, 'seed':seed, 'batch_size': batch_size, 'dropout': 0.25, 'base_lr': base_lr}
    configuration_dict = task.connect(configuration_dict)  # enabling configuration override by clearml
    print(configuration_dict)  # printing actual configuration (after override in remote mode)
    MODEL_STORAGE = f"/mnt/tank/scratch/pgrinkevich/models/{task.task_id}"
    if not os.path.exists(MODEL_STORAGE):
        os.mkdir(MODEL_STORAGE) # create model storage
    logger = task.get_logger()

    # Data
    df_train = pd.read_csv("deepfashion_train.csv")
    df_val = pd.read_csv("deepfashion_val.csv")
    num_classes_train = len(df_train.label.unique()) # reassign class numbers so it doesn't go over num_classes_train 
    df_train['label'] = df_train.groupby(['label']).ngroup()
    if opt.model == "enet":
        img_size = 256
    elif opt.model == "dummy":
        img_size = 256


    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_dataset = DeepFashionDataset(df_train, transform=train_transform)
    val_dataset = DeepFashionDataset(df_val, transform=val_transform)

    # Split Data
    evens = list(range(0, len(val_dataset), 2))
    odds = list(range(1, len(val_dataset), 2))
    query_dataset = Subset(val_dataset, evens)
    retrieval_dataset = Subset(val_dataset, odds)
    # Dataloaders
    iterations_in_epoch = math.ceil(len(train_dataset) / batch_size)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=8
    )
    # Model
    model = get_model(opt.model).to(device)
    #model = torch.load("/mnt/tank/scratch/pgrinkevich/models/c37a6274ca8c4cd9b3bdf66136a408f0/model_ckpt_epoch_40.pth").to(device)
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    loss_func = losses.SubCenterArcFaceLoss(num_classes=num_classes_train, embedding_size=512).to(device)
    loss_optimizer = torch.optim.Adam(loss_func.parameters(), lr=1e-4)
    accuracy_calculator = AccuracyCalculator(include=("precision_at_1","mean_average_precision",), k="max_bin_count", device=device)
    # Start training
    for epoch in range(0, num_epochs):
        train(model, loss_func, device, train_loader, optimizer, loss_optimizer, epoch, iterations_in_epoch, logger)
        test(query_dataset, retrieval_dataset, model, accuracy_calculator, epoch, iterations_in_epoch, logger)
        torch.save(model, f'{MODEL_STORAGE}/model_ckpt_epoch_{epoch}.pth')


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='dummy', help='model architecture')
    parser.add_argument('--task_name', type=str, default='EffNet', help='ClearML task name')

    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)



#srun --cpus-per-task=20 -p aihub --gres=gpu:3 --mem=20G --time=20:00:00 --pty bash
#python3 ./metric_learning/training_script.py --model enet --task_name effnetv2s