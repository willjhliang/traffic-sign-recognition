
import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
import itertools


def center_crop(img, center_percentage):
    """Crops out edges of an image, leaving the center."""
    width, height, _ = img.shape
    width_offset = int(width * (1 - center_percentage) / 2)
    height_offset = int(height * (1 - center_percentage) / 2)
    img = img[width_offset:width-width_offset, height_offset:height-height_offset]
    return img


def rotate_img(img, angle):
    """Rotates an image and replaces empty space with black."""
    height, width, _ = img.shape
    center_x, center_y = (width // 2, height // 2)

    rot_mat = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    cos = np.abs(rot_mat[0, 0])
    sin = np.abs(rot_mat[0, 1])

    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    rot_mat[0, 2] += (new_width / 2) - center_x
    rot_mat[1, 2] += (new_height / 2) - center_y

    img = cv2.warpAffine(img, rot_mat, (new_width, new_height))
    img = cv2.resize(img, (width, height))

    return img


def shift_brightness(img, shift):
    """Adjusts brightness of all pixels in image."""
    img = np.clip(img + shift, 0, 1)
    return img


def augment_img(img):
    """Augments image with rotation, cropping, and brightness shifts."""
    rot_angle = random.randint(-20, 20)
    crop_center_percentage = random.randint(70, 90) / 100
    crop_center_percentage = 0.8
    brightness_shift = random.randint(-10, 10) / 100

    img = rotate_img(img, rot_angle)
    # img = center_crop(img, crop_center_percentage)
    # img = shift_brightness(img, brightness_shift)
    img = center_crop(img, 0.8)

    return img


def augment_dataset(train_data):
    """Applies augmentation to all classes in dataset."""
    max_k_size = max([len(train_data[k]) for k in range(len(train_data))])
    for k in range(len(train_data)):
        k_size = len(train_data[k])
        for i in range(max_k_size - k_size):  # Add augmented images until we have class_size images
            train_data[k].append(augment_img(train_data[k][i % k_size]))
    return train_data


def visualize_augmentation(train_data):
    """Visualizes the augmentation applied onto the last image in each class."""
    fig, axs = plt.subplots(4, 10, figsize=(15, 5))
    for k, (i, j) in itertools.zip_longest(range(len(train_data)), list(itertools.product(range(4), range(10))), fillvalue=-1):
        axs[i,j].axis('off')
        if k >= 0:
            img = augment_img(train_data[k][-1])
            axs[i,j].imshow(augment_img(img))
    plt.show()