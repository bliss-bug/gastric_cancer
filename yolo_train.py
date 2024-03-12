import torch
import argparse
from ultralytics import YOLO


def main(args):
    model = YOLO(args.weight_path)
    model.train(data="dataset/data.yaml", epochs=args.epochs, imgsz=args.image_size, 
                device=args.device_ids, batch=args.batch_size, optimizer='AdamW',
                lr0=args.lr, iou=args.iou)
    
    results = model(['/data/ssd/zhujh/gastric_data/second/images/IMG_01.0000001047354.24084.0044.16555900669.jpg',
                     '/data/ssd/zhujh/gastric_data/second/images/IMG_01.0000001042312.24114.0060.10322500783.jpg',
                     '/data/ssd/zhujh/gastric_data/first/images/IMG_01.0000000433634.0027.5845160.jpg'])
    for result in results:
        print(result.boxes.xywhn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--iou', type=float, default=0.4)

    parser.add_argument('--device_ids', nargs='+', type=int, default=[0,1])
    parser.add_argument('--image_size', type=int, default=640)
    parser.add_argument('--weight_path', type=str, default='yolov8n.pt')

    args = parser.parse_args()

    main(args)