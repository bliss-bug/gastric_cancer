import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models import ResNet50_Weights

from dataset import MyDataset
from utils import *
from model import ResUNet


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.model == 'deeplabv3_resnet50':
        model = deeplabv3_resnet50(num_classes=args.num_classes, backbone=ResNet50_Weights).to(device)
    elif args.model == 'ResUNet':
        model = ResUNet(in_ch=3, n_class=args.num_classes).to(device)
    model = nn.DataParallel(model, device_ids=args.device_ids)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_sch = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

    train_path, train_masks, _ = load_data(args.paths, 'train')
    val_path, val_masks, _ = load_data(args.paths, 'val')
    test_path, test_masks, _ = load_data(args.paths, 'test')

    train_dataset = MyDataset(train_path, train_masks, args.image_size)
    val_dataset = MyDataset(val_path, val_masks, args.image_size)
    test_dataset = MyDataset(test_path, test_masks, args.image_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=8, pin_memory=True)
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=8, pin_memory=True)
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=8, pin_memory=True)

    miou_max = 0

    for i in range(args.epochs):
        loss = train_one_epoch(model, optimizer, lr_sch, train_loader, device, args.model)
        print('train {}: loss: {:.3f}\n'.format(i+1, loss))
        with open('outcome/{}.log'.format(args.model), 'a') as file:
            file.write('train {}: loss: {:.3f}\n'.format(i+1, loss))

        miou, dsc, miou2, dsc2 = val_one_epoch(model, val_loader, device, args.model, args.num_classes)

        if miou2 > miou_max:
            torch.save(model.module.state_dict(), "weight/{}.pth".format(args.model))
            miou_max = miou2

        print('val {}: mIoU: {:.3f}, DSC: {:.3f}, mIoU2: {:.3f}, DSC2: {:.3f}\n'.format(i+1, miou, dsc, miou2, dsc2))
        with open('outcome/{}.log'.format(args.model), 'a') as file:
            file.write('val {}: mIoU: {:.3f}, DSC: {:.3f}, mIoU2: {:.3f}, DSC2: {:.3f}\n'.format(i+1, miou, dsc, miou2, dsc2))


    if args.model == 'deeplabv3_resnet50':
        model = deeplabv3_resnet50(num_classes=args.num_classes, backbone=ResNet50_Weights).to(device)
    elif args.model == 'ResUNet':
        model = ResUNet(in_ch=3, n_class=args.num_classes).to(device)
        
    model.load_state_dict(torch.load("weight/{}.pth".format(args.model)))
    model = nn.DataParallel(model, device_ids=args.device_ids)

    miou_test, dsc_test, miou2_test, dsc2_test = val_one_epoch(model, test_loader, device, args.model, args.num_classes)
    print('test: mIoU: {:.3f}, DSC: {:.3f}, mIoU2: {:.3f}, DSC2: {:.3f}\n'.format(miou_test, dsc_test, miou2_test, dsc2_test))

    with open('outcome/{}.log'.format(args.model), 'a') as file:
        file.write('test: mIoU: {:.3f}, DSC: {:.3f}, mIoU2: {:.3f}, DSC2: {:.3f}\n\n'.format(miou_test, dsc_test, miou2_test, dsc2_test))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--device_ids', nargs='+', type=int, default=[2,3])
    parser.add_argument('--image_size', type=int, default=512, help="target input image size")
    parser.add_argument('--model', type=str, default='deeplabv3_resnet50')
    parser.add_argument('--paths', type=list, default=['/data/ssd/zhujh/gastric_data/first/', '/data/ssd/zhujh/gastric_data/second/'])

    args = parser.parse_args()

    main(args)