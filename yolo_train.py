import torch
import argparse
from ultralytics import YOLO


def main(args):
    model = YOLO(args.weight_path)

    results = model.train(data="dataset/data.yaml", epochs=args.epochs, imgsz=args.image_size, 
                device=args.device_ids, batch=args.batch_size, optimizer='Adam')  # train the model
    print(results)
    #metrics = model.val()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--device_ids', nargs='+', type=int, default=[0,1])
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--weight_path', type=str, default='yolov8n.pt')

    args = parser.parse_args()

    main(args)