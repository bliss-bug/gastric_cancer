import torch
import argparse
from ultralytics import YOLO


def main(args):
    model = YOLO("yolov8n.pt")

    results = model.train(data="data.yaml", epochs=args.epochs, imgsz=args.image_size, 
                device=args.device_ids, batch=args.batch_size, optimizer='AdamW',
                cache=True)  # train the model
    #metrics = model.val()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--device_ids', nargs='+', type=int, default=[0,1])
    parser.add_argument('--image_size', type=int, default=512)

    args = parser.parse_args()

    main(args)