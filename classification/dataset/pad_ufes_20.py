import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Any, Optional, Tuple
from PIL.Image import open as open_image

class PadUfes20(Dataset):
    def __init__(self, root: str, train: bool = True, transform: Optional[transforms.Compose] = None):
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])
            
        self.transform = transform
        self.train = train
        self.root = root
        
        # Defining .csv file to be used
        if self.train:
            csv_name = "pad-ufes-20_parsed_folders.csv"
        else:
            csv_name = "pad-ufes-20_parsed_test.csv"
            
        # Opening dataframe:
        self.df = pd.read_csv(os.path.join(self.root, csv_name), header = 0)
        self.x = "img_id"
        self.y = "diagnostic_number"
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        if torch.is_tensor(index):
            index = index.tolist()
                
        img_path = os.path.join(self.root, "images", self.df.loc[index][self.x])
        img = open_image(img_path)
        
        img = self.transform(img)
            
        return img, int(self.df.loc[index][self.y])