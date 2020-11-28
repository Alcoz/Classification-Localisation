from skimage import transform
import cv2
import torch
import numpy as np
class Rescale(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, bndbox, label = sample['image'], sample['bndbox'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_h, new_w))

        bndbox = np.array([int(bndbox[0] * new_w / w), int(bndbox[1] * new_h / h), int(bndbox[2] * new_w / w), int(bndbox[3] * new_h / h)])

        return {'image': img, 'bndbox': bndbox, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, bndbox, label = sample['image'], sample['bndbox'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'bndbox': torch.from_numpy(bndbox),
                'label': label}
