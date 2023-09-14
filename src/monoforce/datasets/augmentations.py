import numpy as np


# data augmentation
def horizontal_shift(img, shift):
    if shift > 0:
        img_shifted = np.zeros_like(img)
        img_shifted[..., :shift] = img[..., -shift:]
        img_shifted[..., shift:] = img[..., :-shift]
    else:
        img_shifted = img
    return img_shifted
