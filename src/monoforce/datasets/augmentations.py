import numpy as np
from torchvision import transforms
import PIL.Image as pil


__all__ = [
    'horizontal_shift',
]

# data augmentation
def horizontal_shift(img, shift):
    assert shift < img.shape[1]
    assert shift > -img.shape[1]
    assert img.ndim == 2 or img.ndim == 3
    if shift > 0:
        img_shifted = np.zeros_like(img)
        img_shifted[:, :shift] = img[:, -shift:]
        img_shifted[:, shift:] = img[:, :-shift]
    else:
        img_shifted = img
    return img_shifted


def demo():
    import matplotlib.pyplot as plt
    from .data import DEMPathData

    ds = DEMPathData('/home/ruslan/data/robingas/data/22-08-12-cimicky_haj/marv/ugv_2022-08-12-15-18-34_trav/')
    # test augmentation
    img = ds.get_raw_image(0, 'front')
    img = pil.fromarray(img)

    color_aug = transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=(-0.1, 0.1))
    print(color_aug)
    # apply augmentation
    img_aug = color_aug(img)

    shift = 500
    img_shifted = horizontal_shift(np.asarray(img), shift)

    # visualize
    plt.figure()
    plt.subplot(131)
    plt.imshow(img)
    plt.subplot(132)
    plt.imshow(img_aug)
    plt.subplot(133)
    plt.imshow(img_shifted)
    plt.show()


if __name__ == '__main__':
    demo()
