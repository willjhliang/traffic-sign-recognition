import os
import numpy as np
from PIL import Image
import csv

K = 36


def create_small_dataset(datapath):
    """Loads images from files and performs basic pre-processing."""
    data = {}
    with open('data/labels.csv') as f:
        reader = csv.DictReader(f)
        labels = list(reader)
    new_labels = []
    for k in range(K):
        data[k] = []


    class_count = [0] * 36

    os.mkdir('data/filtered_images/train_small')

    for f in os.listdir(datapath):
        k = int(f[:3])  # Get label from filename
        if class_count[k] < 10:
            img = Image.open(os.path.join(datapath, f))
            img.save(f"data/filtered_images/train_small/{f}")
            class_count[k] = class_count[k] + 1


create_small_dataset('data/filtered_images/train')
