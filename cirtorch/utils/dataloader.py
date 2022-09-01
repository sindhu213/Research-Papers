import os
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self,root_dir,transforms=None):
        self.root_dir = root_dir
        self.transform = transforms
        self.data = os.listdir(root_dir)

    def __getitem__(self, index):
        img_path =  os.path.join(self.root_dir,self.data[index])
        image = Image.open(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.data)


## might change some features here
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([64,64]),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 64
torch.manual_seed(42)
train = ImageDataset('./cirtorch/data/train/',transform)
valid = ImageDataset('./cirtorch/data/valid/',transform)

train_dl = DataLoader(train,batch_size=batch_size,shuffle=True,drop_last=True)
valid_dl = DataLoader(valid,batch_size=batch_size,shuffle=True,drop_last=True)