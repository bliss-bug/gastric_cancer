import json
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


def load_path():
    with open('images.json') as file:
        images_path = json.load(file)

    with open('tags.json') as file:
        tags_path = json.load(file)

    return images_path, tags_path


if __name__ == '__main__':
    im = np.array(Image.open('/data/ssd/zhujh/gastric_data/second/img/IMG_01.0000001008393.0062.08351300007.jpg'))
    print(im[:,:,0].mean())

    im = np.array(Image.open('/data/ssd/zhujh/gastric_data/first/img/IMG_01.0000000000325.0005.7102287.jpg'))
    print(im[:,:,0].mean())

    im = np.array(Image.open('/data/ssd/zhujh/gastric_data/first/img/IMG_01.0000000000325.0013.7102305.jpg'))
    print(im[:,:,0].mean())

    im = np.array(Image.open('/data/ssd/zhujh/gastric_data/first/img/IMG_01.0000000007729.0020.2097556.jpg'))
    print(im[:,:,0].mean())

    im = np.array(Image.open('/data/ssd/zhujh/gastric_data/first/img/IMG_01.0000000017978.0018.4291315.jpg'))
    print(im[:,:,0].mean())

    print('----------------------------------------------------')

    im = np.array(Image.open('/data/ssd/zhujh/gastric_data/second/img/IMG_01.0000001008393.0064.08361800642.jpg'))
    print(im[:,:,0].mean())

    im = np.array(Image.open('/data/ssd/zhujh/gastric_data/second/img/IMG_01.0000001008393.0065.08363200520.jpg'))
    print(im[:,:,0].mean())

    im = np.array(Image.open('/data/ssd/zhujh/gastric_data/first/img/IMG_01.0000000000325.0006.7102291.jpg'))
    print(im[:,:,0].mean())

    im = np.array(Image.open('/data/ssd/zhujh/gastric_data/first/img/IMG_01.0000000000325.0012.7102303.jpg'))
    print(im[:,:,0].mean())

    #images_path, tags_path = load_path()
    #images = [Image.open(image_path) for image_path in images_path]
    