import json
import os
from tqdm import trange, tqdm
import sys
import random

from PIL import Image, ImageDraw
import cv2
import numpy as np

import torch

from loss import DFLoss
from metrics import *


def load_data(data_paths, mode='train'):
    '''
    images_path, tags_path = [], []

    for data_path in data_paths:
        imgs_path = os.listdir(data_path+'img/')
        tgs_path = os.listdir(data_path+'tag/')
        images_path.extend([os.path.join(data_path+'img/', img_path) for img_path in imgs_path])
        tags_path.extend([os.path.join(data_path+'tag/', tg_path) for tg_path in tgs_path])
    
    images_path.sort()
    tags_path.sort()

    images_tags = list(zip(images_path, tags_path))
    random.shuffle(images_tags)
    images_path = [images_tags[i][0] for i in range(len(images_tags))]
    tags_path = [images_tags[i][1] for i in range(len(images_tags))]

    js_images = json.dumps(images_path)
    js_tags = json.dumps(tags_path)

    with open('images.json', 'w') as file:
        file.write(js_images)
    
    with open('tags.json', 'w') as file:
        file.write(js_tags)
    '''
    with open('images.json') as file:
        images_path = json.load(file)

    with open('tags.json') as file:
        tags_path = json.load(file)

    print('{} data'.format(mode))
    
    n = len(images_path)//1
    if mode == 'train':
        s, t = 0, n*7//10
    elif mode == 'val':
        s, t = n*7//10, n*8//10
    else:
        s, t = n*8//10, n

    masks, all_boxes = [], []
    for i in trange(s, t):
        boxes = []
        if os.path.getsize(tags_path[i]) > 0:
            with open(tags_path[i],'r') as f:
                content = json.load(f)
                lst_len = len(content[0]['pts'])
                types, lsts = [content[0]['pts'][j]['id'] for j in range(lst_len)], [content[0]['pts'][j]['lst'] for j in range(lst_len)]
                H, W = content[0]['height'], content[0]['width']

            mask = np.zeros((H, W), dtype=np.uint8)

            for type, lst in zip(types, lsts):
                poly = np.array([[[round(p['x']), round(p['y'])] for p in lst]], dtype=int)
                cv2.fillPoly(mask, poly, color=1)
                box = [poly[0,:,0].min(), poly[0,:,1].min(), poly[0,:,0].max(), poly[0,:,1].max()]
                boxes.append(box)
        else:
            image = Image.open(images_path[i])
            image_data = np.array(image)
            H, W = image_data.shape[:2]
            mask = np.zeros((H, W), dtype=np.uint8)
            boxes.append([0, 0, W-1, H-1])

        boxes = np.array(boxes)

        masks.append(mask)
        all_boxes.append(boxes)
        
    return images_path[s:t], masks, all_boxes



def draw_polygon(image_path, tag_path):
    image = Image.open(image_path)
    if os.path.getsize(tag_path) > 0:
        with open(tag_path,'r') as f:
            content = json.load(f)
            lst_len = len(content[0]['pts'])
            types, lsts = [content[0]['pts'][j]['id'] for j in range(lst_len)], [content[0]['pts'][j]['lst'] for j in range(lst_len)]

        draw = ImageDraw.Draw(image)
        for type, lst in list(zip(types, lsts)):
            for j in range(-1, len(lst)-1):
                line = [(lst[j]['x'], lst[j]['y']), (lst[j+1]['x'], lst[j+1]['y'])]
                if type == 0:
                    draw.line(line, fill='red', width=2)
                else:
                    draw.line(line, fill='blue', width=2)
        
    image.save('image.jpg')
    image.close()



def draw_mask(image_path, tag_path):
    if os.path.getsize(tag_path) > 0:
        with open(tag_path,'r') as f:
            content = json.load(f)
            lst_len = len(content[0]['pts'])
            types, lsts = [content[0]['pts'][j]['id'] for j in range(lst_len)], [content[0]['pts'][j]['lst'] for j in range(lst_len)]
            H, W = content[0]['height'], content[0]['width']

        mask = np.zeros((H, W), dtype=np.uint8)
        for j, lst in enumerate(lsts):
            poly = np.array([[[round(p['x']), round(p['y'])] for p in lst]], dtype=np.int32)
            cv2.fillPoly(mask, poly, color=types[j]+1)

    else:
        image = Image.open(image_path)
        image_data = np.array(image)
        H, W = image_data.shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)

    image_mask = Image.fromarray(mask*100)
    image_mask.save('mask.jpg')
    image_mask.close()



def train_one_epoch(model, optimizer, lr_sch, data_loader, device, model_name):
    model.train()
    loss_func = DFLoss(lamda=[1,3])
    losses, cnt = 0., 0

    data_loader = tqdm(data_loader, file=sys.stdout)
    for images, labels, _ in data_loader:
        optimizer.zero_grad()
        images, labels = images.to(device), labels.to(device).long()  # [N, 3, H, W], [N, H, W]
        N = images.shape[0]
        
        if model_name == 'deeplabv3_resnet50':
            pred = model(images)['out']  # [N, C, H, W]
        else:
            pred = model(images)

        loss = loss_func(pred, labels)
        losses += loss.item() * N
        cnt += N
        loss.backward()
        optimizer.step()
    
    lr_sch.step()

    return losses / cnt



def val_one_epoch(model, data_loader, device, model_name, n_classes=3):
    model.eval()
    miou, dsc, num_samples = 0, 0, 0
    cm = torch.zeros((n_classes, n_classes), dtype=torch.float64).to(device)

    with torch.no_grad():
        data_loader = tqdm(data_loader, file=sys.stdout)
        for images, labels, _ in data_loader:
            images, labels = images.to(device), labels.to(device).long()

            if model_name == 'deeplabv3_resnet50':
                pred = model(images)['out'].softmax(dim=1)
            else:
                pred = model(images).softmax(dim=1)
            pred_class = pred.argmax(dim=1)  # [N, H, W]

            num_samples += images.shape[0]
            miou += mIoU(pred_class, labels, 3) * images.shape[0]
            dsc += Dice(pred_class, labels, 3) * images.shape[0]

            cm += torch.bincount(labels.view(-1)*n_classes + pred_class.view(-1), minlength=n_classes**2).view(n_classes, n_classes).to(torch.float64)

    miou /= num_samples
    dsc /= num_samples

    ious = cm.diag() / (cm.sum(dim=0) + cm.sum(dim=1) - cm.diag())
    miou2 = ious.mean()

    dscs = 2.*cm.diag() / (cm.sum(dim=0) + cm.sum(dim=1))
    dsc2 = dscs.mean()

    return miou.item(), dsc.item(), miou2.item(), dsc2.item()



if __name__ == '__main__':
    load_data(None, 'train')
    load_data(None, 'val')
    load_data(None, 'test')