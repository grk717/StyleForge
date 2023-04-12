import torch
import ffapi
import datetime
import os
import pandas as pd
from PIL import Image
import requests
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from itertools import repeat
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np

DATABASE_IMAGES_DIR = "./database/database_images"
DOWNLOAD_IMAGES = False
BATCH_SIZE = 32
date = datetime.datetime.today().strftime('%m-%d-%y')

class FarFetchDataset(Dataset):
    def __init__(self, image_dir, transform):
        image_names = os.listdir(image_dir)
        self.images = [os.path.join(image_dir, i) for i in image_names]
        self.ids = [int(i.split("_")[0]) for i in image_names]
        self.types = [i.split("_")[1].split('.')[0] for i in image_names]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        return self.transform(image), self.ids[idx], self.types[idx]
    
def download_url(*args):
    image_id, image_url, image_type = args[0]
    image = Image.open(requests.get(image_url, stream=True).raw)
    image.save(os.path.join(DATABASE_IMAGES_DIR, f"{image_id}_{image_type}.jpg"))

def download_images(df):
    if not os.path.exists(DATABASE_IMAGES_DIR):
        os.mkdir(DATABASE_IMAGES_DIR)
    cpus = cpu_count()
    results = ThreadPool(cpus - 1).map_async(download_url, list(zip(df['id'], df["images.model"], repeat("model", len(df)))))
    results.wait()
    
def add_embeddings(df):
    inference_transforms = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = FarFetchDataset(DATABASE_IMAGES_DIR, inference_transforms)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model = torch.load("./database/epoch99.pth").cuda()

    df['embeddings_model'] = np.nan
    df['embeddings_model'] = df['embeddings_model'].astype(object)
    df = df.loc[:, ['embeddings_model']]
    for samples in tqdm(dataloader):
        images, ids, types = samples
        images = images.cuda()
        with torch.no_grad():
            embeddings = model(images).cpu().numpy()
            temp_series = pd.Series(embeddings.tolist())
            df.loc[ids, "embeddings_model"] = temp_series.values
    
    df.to_pickle(f"with_embeddings{date}.pkl")

#FILENAME = 'current_farfetch_listings' + date + '.csv'
FILENAME = r"./database/current_farfetch_listings04-09-23.csv"
if not os.path.exists(FILENAME):
    api = ffapi.Api()
    total_pages = api.get_listings()['listingPagination']['totalPages']
    

    def create_database():
        for page in tqdm(range(1, total_pages)):
            api.parse_products(
                api.get_listings(page=page)
            )
        return api.df


    create_database()
    api.df.to_csv(FILENAME)


df = pd.read_csv(FILENAME)
print(df.columns)


if DOWNLOAD_IMAGES:
    download_images(df)
if __name__ == "__main__":
    df = pd.read_csv(FILENAME)
    df = df.set_index("id")
    add_embeddings(df)