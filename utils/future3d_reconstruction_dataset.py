import os
import os
import json
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset

class Future3D_Reconstruction_Dataset(Dataset):
    def __init__(self, data_root, train_set_json='train_set.json', image_size=224, transform=None):
        self.data_root = data_root
        self.image_size = image_size
        with open(os.path.join(self.data_root, 'data_info', train_set_json)) as f:
            self.train_set_info = json.load(f)
        self.transform = transform
        
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor()
            ])

    def __getitem__(self, index):
        item_info = self.train_set_info[index]
        img = Image.open(os.path.join(self.data_root, 'image', item_info['image'])).convert('RGB')
        mask = Image.open(os.path.join(self.data_root, 'mask', item_info['mask']))
        
        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)
            
        output = {
            'image': img,
            'mask': mask,
            'model': item_info['model'],
            'texture': item_info['texture'],
            'translation': torch.tensor(item_info['pose']['translation']).float(),
            'rotation': torch.tensor(item_info['pose']['rotation']).float(),
            'fov': torch.tensor(item_info['fov']).float()
        }
        
        return output

    def __len__(self):
        return len(self.train_set_info)

