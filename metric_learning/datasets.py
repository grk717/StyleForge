import pandas as pd
import json
import os
from tqdm import tqdm
from random import Random

def convert_dataset(ann_dir, output_path):
    columns = ["id", "image_path", "pair_id", "style", "x_1", "y_1", "x_2", "y_2"]
    df = pd.DataFrame(columns=columns)
    ann_list = [os.path.join(ann_dir, i) for i in os.listdir(ann_dir)]
    for ann_path in tqdm(ann_list):
        with open(ann_path, 'r') as ann_file:
            ann = json.load(ann_file)
        # count number of items in annotation
        noi = len([key for key in list(ann.keys()) if key.startswith("item")])
        # for each item in image form a row in df
        for item_num in range(1, noi+1):
            df_row = {key: None for key in columns}
            df_row["id"] = ann_path.split(".")[0].split("\\")[-1]
            df_row["image_path"] = ann_path.replace("annos", "image").split(".")[0] + ".jpg"
            df_row["pair_id"] = ann["pair_id"]
            df_row["style"] = ann["item" + str(item_num)]["style"]
            bbox = ann["item" + str(item_num)]["bounding_box"]
            df_row["x_1"], df_row["y_1"], df_row["x_2"], df_row["y_2"] = bbox
            df_row = pd.DataFrame([df_row])
            df = pd.concat([df, df_row], ignore_index=True)
    # just checkpoint
    df_pairs = df[df['style'] != 0]
    # create label column
    df_pairs['label'] = df_pairs.groupby(['pair_id', 'style']).ngroup()
    # save df
    df_pairs.to_csv(output_path, index=False)  


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