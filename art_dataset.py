import torch
from PIL import Image
from dotenv import load_dotenv
import torchvision.transforms as T
import os


class AbstractArtDataset(torch.utils.data.Dataset):
    def __init__(self, list_ids):
        self.list_ids = list_ids

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        id = self.list_ids[index]
        load_dotenv('.env')
        root = os.getenv("root")
        sample = Image.open(root+"Abstract_image_"+str(id)+".jpg")

        composed_transforms = T.Compose([T.Resize(
            size=(1024, 1024)), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        sample = composed_transforms(sample)

        return sample
