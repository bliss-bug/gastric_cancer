import sys
from PIL import Image
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from segment_anything import sam_model_registry

from utils import load_data
from dataset import SamDataset
from loss import DFLoss
from model import MedSAM


def main():
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--device_ids', nargs='+', type=int, default=[2,3,4,5,6,7])
    parser.add_argument('--image_size', type=int, default=512, help="target input image size")
    parser.add_argument('--paths', type=list, default=['data/first/', 'data/second/'])
    parser.add_argument('--weight_path', type=str, default='/data/ssd/zhujh/medsam_vit_b.pth')

    args = parser.parse_args()

    main(args)