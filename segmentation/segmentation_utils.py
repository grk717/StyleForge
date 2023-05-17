import albumentations as albu
import pandas as pd
import json
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np



def convert_dataset(ann_dir, output_path):
    """
    Creation of a segmentation masks and pandas DataFrame from DeepFashion 2 dataset
    """
    columns = ["id", "image_path", "pair_id", "style", "x_1", "y_1", "x_2", "y_2", "mask_path", "crop_path"]
    df = pd.DataFrame(columns=columns)
    ann_list = [os.path.join(ann_dir, i) for i in os.listdir(ann_dir)]
    for ann_path in tqdm(ann_list):
        with open(ann_path, 'r') as ann_file:
            ann = json.load(ann_file)
        # count number of items in annotation
        noi = len([key for key in list(ann.keys()) if key.startswith("item")])
        # for each item in image form a row in df
        image_path = ann_path.replace("annos", "image").split(".")[0] + ".jpg"
        image = cv2.imread(image_path)
        mask = np.zeros(image.shape[:2])
        for item_num in range(1, noi+1):
            polys = ann["item" + str(item_num)]["segmentation"]
            for poly in polys:
            # new contour coords + clip results
                contour = np.array(poly).reshape(-1, 2).astype('int32')
                cv2.fillPoly(mask, pts=[contour], color=ann["item" + str(item_num)]["category_id"])
        for item_num in range(1, noi+1):
            df_row = {key: None for key in columns}
            df_row["id"] = ann_path.split(".")[0].split("\\")[-1]
            df_row["image_path"] = image_path
            df_row["pair_id"] = ann["pair_id"]
            df_row["style"] = ann["item" + str(item_num)]["style"]
            df_row['category_id'] = ann["item" + str(item_num)]["category_id"]
            df_row['category_name'] = ann["item" + str(item_num)]["category_name"]
            bbox = ann["item" + str(item_num)]["bounding_box"]
            df_row["x_1"], df_row["y_1"], df_row["x_2"], df_row["y_2"] = bbox

            crop_path = ann_path.replace("annos", "crops").split(".")[0] + f"_{item_num}.jpg"
            mask_path = crop_path.replace("crops", "masks").split(".")[0] + ".png"
            # crop and save
            img_crop = cv2.imread(df_row["image_path"])[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            mask_crop = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            cv2.imwrite(crop_path, img_crop)
            cv2.imwrite(mask_path, mask_crop)
            # mask from crop and save
            df_row['mask_path'] = mask_path
            df_row['crop_path'] = crop_path
            df_row = pd.DataFrame([df_row])
            df = pd.concat([df, df_row], ignore_index=True)
    # save df
    df.to_csv(output_path, index=False)  


class DeepFashionDataset(Dataset):
    def __init__(self, df, classes, transform=None, preprocessing=None):
        self.df = df[:50]
        self.classes = classes
        self.transform = transform
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[[idx]]
        image = cv2.imread(row['crop_path'].values[0], cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(row['mask_path'].values[0], cv2.IMREAD_UNCHANGED)
        masks = [(mask == v) for v in self.classes]
        mask = np.stack(masks, axis=-1).astype('float')
        if self.transform:
            sample = self.transform(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        return image, mask

def collate_fn(data):
    images, masks = list(zip(*data))
    images = torch.tensor(np.array(images))
    masks = torch.tensor(np.array(masks)).long()
    return images, masks

def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(320, 320, always_apply=True, border_mode=cv2.BORDER_CONSTANT),
        albu.CenterCrop(320, 320, always_apply=True)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)