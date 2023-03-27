from torch.utils.data import Dataset
from PIL import Image


class DeepFashionDataset(Dataset):
    def __init__(self, df, transform=None, target_transform=None):
        self.df = df
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[[idx]]
        image = Image.open(row['image_path'].values[0])
        bbox = row['x_1'].values[0], row['x_2'].values[0], row['y_1'].values[0], row['y_2'].values[0]
        cropped_image = image.crop((bbox[0], bbox[2], bbox[1], bbox[3]))
        label = row['label'].values[0]
        if self.transform:
            cropped_image = self.transform(cropped_image)
        if self.target_transform:
            label = self.target_transform(label)
        return cropped_image, label