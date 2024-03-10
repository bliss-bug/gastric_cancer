import argparse
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50

from dataset import MyDataset
from utils import *
from model import ResUNet


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.model == 'deeplabv3_resnet50':
        model = deeplabv3_resnet50(num_classes=args.num_classes).to(device)
    elif args.model == 'ResUNet':
        model = ResUNet(in_ch=3, n_class=args.num_classes).to(device)

    model.load_state_dict(torch.load("weight/{}.pth".format(args.model)))
    model = nn.DataParallel(model, device_ids=[0,5,6,7])

    test_path, test_masks, test_boxes = load_data(args.paths, 'test')

    test_dataset = MyDataset(test_path, test_masks, args.image_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=16, pin_memory=True)

    model.eval()
    cnt1, cnt2 = 0, 0
    with torch.no_grad():
        for i, (images, labels, paths) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device).long()

            if args.model == 'deeplabv3_resnet50':
                pred = model(images)['out'].softmax(dim=1)
            else:
                pred = model(images).softmax(dim=1)  # [N, C, H, W]
            pred_class = pred.argmax(dim=1)  # [N, H, W]

            #print('1:', (pred_class==1).sum(), (labels==1).sum())
            #print('2:', (pred_class==2).sum(), (labels==2).sum())
            
            for j in range(images.shape[0]):
                if 1 in labels[j] and 1 not in pred_class[j]:
                    cnt1 += 1
                    pred_class *= 100
                    labels *= 100

                    p, l = pred_class[j].cpu().numpy().astype(np.uint8), labels[j].cpu().numpy().astype(np.uint8)
                    p, l = Image.fromarray(p), Image.fromarray(l)
                    p.save('temp1/'+str(cnt1)+'_predict'+'.jpg')
                    l.save('temp1/'+str(cnt1)+'_label'+'.jpg')     
                
                elif 2 in labels[j] and 2 not in pred_class[j]:
                    cnt2 += 1
                    pred_class *= 100
                    labels *= 100

                    p, l = pred_class[j].cpu().numpy().astype(np.uint8), labels[j].cpu().numpy().astype(np.uint8)
                    p, l = Image.fromarray(p), Image.fromarray(l)
                    p.save('temp2/'+str(cnt2)+'_predict'+'.jpg')
                    l.save('temp2/'+str(cnt2)+'_label'+'.jpg')  
      


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=512)

    parser.add_argument('--device', type=str, default='cuda:6')
    parser.add_argument('--model', type=str, default='deeplabv3_resnet50')
    parser.add_argument('--paths', type=list, default=['/data/ssd/zhujh/gastric_data/second/'])

    args = parser.parse_args()
    main(args)