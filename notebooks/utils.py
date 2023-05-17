import pandas as pd
import json
import os
from tqdm import tqdm
from random import Random
import cv2
import numpy as np

import cv2
import numpy as np
def convert_dataset(ann_dir, output_path):
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



def split_train_test(df: pd.DataFrame, train_classes_frac: float, seed: int=123):
    # get unique labels
    unique_labels = df.label.unique()
    # split to train / val 
    train_classes_len = int(len(unique_labels) * train_classes_frac)

    rng = Random(seed)
    rng.shuffle(unique_labels)

    train_classes = unique_labels[:train_classes_len]
    val_classes = unique_labels[train_classes_len:]

    return df[df.label.isin(train_classes)], df[df.label.isin(val_classes)]