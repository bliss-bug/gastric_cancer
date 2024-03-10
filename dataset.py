import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms, InterpolationMode

from PIL import Image, ImageDraw
import numpy as np


class MyDataset(Dataset):
    def __init__(self, images_path:list, masks:list, image_size=512):
        self.images_path = images_path
        self.masks = masks

        self.image_transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        self.mask_transform = transforms.Compose([transforms.Resize((image_size, image_size), InterpolationMode.NEAREST)])                                 

    def __getitem__(self, idx):
        image = Image.open(self.images_path[idx])
        image = self.image_transform(image)

        mask = self.mask_transform(Image.fromarray(self.masks[idx]))
        mask = torch.tensor(np.array(mask), dtype=torch.int)

        return image, mask, self.images_path[idx]

    def __len__(self):
        return len(self.images_path)
    


class SamDataset(Dataset):
    def __init__(self, images_path:list, masks:list, all_boxes:list, image_size=512):
        self.images_path = images_path
        self.masks = masks
        self.all_boxes = all_boxes

        self.image_transform = transforms.Resize((image_size, image_size))
        self.mask_transform = transforms.Compose([transforms.Resize((image_size, image_size), InterpolationMode.NEAREST)])

    def __getitem__(self, idx):
        original_h, original_w = self.masks[idx].shape
        image = Image.open(self.images_path[idx])
        image = self.image_transform(image)
        image = torch.tensor(np.array(image), dtype=torch.float).permute(2, 0, 1)
        h, w = image.shape[-2:]

        mask = self.mask_transform(Image.fromarray(self.masks[idx]))
        mask = torch.tensor(np.array(mask), dtype=torch.int)

        boxes = self.all_boxes[idx]
        box = np.array([10000,10000,0,0], dtype=int)
        for b in boxes:
            box[0], box[1] = min(box[0], b[0]), min(box[1], b[1])
            box[2], box[3] = max(box[2], b[2]), max(box[3], b[3])
        box = torch.tensor(box, dtype=torch.int)
        box[0], box[1] = box[0]*w/original_w+0.5, box[1]*h/original_h+0.5
        box[2], box[3] = box[2]*w/original_w+0.5, box[3]*h/original_h+0.5
        box = box.to(torch.float)

        return image, mask, box, self.images_path[idx]
        
    def __len__(self):
        return len(self.images_path)