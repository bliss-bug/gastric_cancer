import sys
from PIL import Image
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from segment_anything import sam_model_registry

from utils import load_data
from dataset import SamDataset
from loss import DFLoss, DFLoss2
from model import GasSAM


def train(data_loader, model: nn.DataParallel, optimizer, loss_fn, device):
    model.train()
    losses, cnt = 0., 0
    
    data_loader = tqdm(data_loader, file=sys.stdout)
    for images, labels, boxes, _ in data_loader:
        optimizer.zero_grad()
        images, labels, boxes = images.to(device), labels.long().to(device), boxes.to(device)  # [N, 3, H, W], [N, H, W], [N, 4]
        N = images.shape[0]
        pred_masks = model(images, boxes)
        
        loss = loss_fn(pred_masks, labels)
        losses += loss.item() * N
        cnt += N
        loss.backward()
        optimizer.step()

    return losses / cnt



def val(data_loader, model: nn.DataParallel, n_classes, device):
    model.eval()
    cm = torch.zeros((n_classes, n_classes), dtype=torch.float64).to(device)
    
    with torch.no_grad():
        data_loader = tqdm(data_loader, file=sys.stdout)
        for images, labels, boxes, _ in data_loader:
            images, labels, boxes = images.to(device), labels.to(device).long(), boxes.to(device)
            pred_masks = model(images, boxes) # [B, 1, H, W]

            pred_masks = (pred_masks.sigmoid()>0.5).long().squeeze()
            cm += torch.bincount(labels.flatten()*n_classes + pred_masks.flatten(), minlength=n_classes**2).view(n_classes, n_classes).to(torch.float64)

    ious = cm.diag() / (cm.sum(dim=0) + cm.sum(dim=1) - cm.diag())
    miou = ious.mean()

    dscs = 2.*cm.diag() / (cm.sum(dim=0) + cm.sum(dim=1))
    dsc = dscs.mean()

    return miou.mean(), dsc.mean()



def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    train_path, train_masks, train_boxes = load_data(args.paths, 'train')
    val_path, val_masks, val_boxes = load_data(args.paths, 'val')
    test_path, test_masks, test_boxes = load_data(args.paths, 'test')

    train_dataset = SamDataset(train_path, train_masks, train_boxes, args.image_size)
    val_dataset = SamDataset(val_path, val_masks, val_boxes, args.image_size)
    test_dataset = SamDataset(test_path, test_masks, test_boxes, args.image_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=8, pin_memory=True)
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=8, pin_memory=True)
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=8, pin_memory=True)

    sam_model = sam_model_registry['vit_b'](checkpoint=None, image_size=args.image_size).to(device)
    pretrained_state_dict = torch.load(args.weight_path)
    model_state_dict = sam_model.state_dict()
    for name, param in pretrained_state_dict.items():
        if name in model_state_dict and model_state_dict[name].shape == param.shape:
            model_state_dict[name] = param

    sam_model.load_state_dict(model_state_dict)

    # train
    model = GasSAM(sam_model).to(device)
    #model.load_state_dict(torch.load('weight/sam_vit_b.pth'))
    optimizer = optim.AdamW(model.sam_model.mask_decoder.parameters(), lr=args.lr, weight_decay=2e-4)
    model = nn.DataParallel(model, device_ids=args.device_ids)

    loss_fn = DFLoss2([1,3])

    miou_max = 0
    for epoch in range(args.epochs):
        loss = train(train_loader, model, optimizer, loss_fn, device)
        print('train {}: loss: {:.3f}\n'.format(epoch+1, loss))
        with open('outcome/sam_vit_b.log', 'a') as file:
            file.write('train {}: loss: {:.3f}\n'.format(epoch+1, loss))

        miou, dsc = val(val_loader, model, args.num_classes, device)
        if miou > miou_max:
            miou_max = miou
            torch.save(model.module.state_dict(), "weight/sam_vit_b.pth")

        print('val {}: mIoU: {:.3f}, DSC: {:.3f}\n'.format(epoch+1, miou, dsc))
        with open('outcome/sam_vit_b.log', 'a') as file:
            file.write('val {}: mIoU: {:.3f}, DSC: {:.3f}\n'.format(epoch+1, miou, dsc))

    # test
    model = GasSAM(sam_model).to(device)
    model.load_state_dict(torch.load('weight/sam_vit_b.pth'))
    model = nn.DataParallel(model, device_ids=args.device_ids)

    miou_test, dsc_test = val(test_loader, model, args.num_classes, device)

    print('test: mIoU: {:.3f}, DSC: {:.3f}\n'.format(miou_test, dsc_test))
    with open('outcome/sam_vit_b.log', 'a') as file:
        file.write('test: mIoU: {:.3f}, DSC: {:.3f}\n\n'.format(miou_test, dsc_test))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--device_ids', nargs='+', type=int, default=[0,1])
    parser.add_argument('--image_size', type=int, default=512, help="target input image size")
    parser.add_argument('--paths', type=list, default=['/data/ssd/zhujh/gastric/first/', '/data/ssd/zhujh/gastric/second/'])
    parser.add_argument('--weight_path', type=str, default='/data/ssd/zhujh/sam_vit_b_01ec64.pth')

    args = parser.parse_args()

    main(args)