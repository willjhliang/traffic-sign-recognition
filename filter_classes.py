
import os
import numpy as np
from PIL import Image
import csv


K = 58

def clean_data(datapath):
    """Loads images from files and performs basic pre-processing."""
    classFreq = []
    data = {}
    with open('data/labels.csv') as f:
        reader = csv.DictReader(f)
        labels = list(reader)
    new_labels = []
    for k in range(K):
        data[k] = []
        classFreq.append(0)

    os.mkdir('data/filtered_images')

    for f in os.listdir(datapath):
        k = int(f[:3])  # Get label from filename
        classFreq[k] = classFreq[k] + 1
        img = Image.open(os.path.join(datapath, f))
        img = np.asarray(img)
        data[k].append(img)

    kp = 0
    for k in range(K):
        if classFreq[k] < 50:
            continue
        for i, img in enumerate(data[k]):
            img = Image.fromarray(img)
            img.save(f"data/filtered_images/{kp:03}_{i:03}.png")
        kp += 1
        new_labels.append(labels[k])


    with open('data/filtered_labels.csv', 'w') as f2:

        writer = csv.writer(f2)

        writer.writerow(['ClassId', 'Name'])
        for i in range(len(new_labels)):
            towrite = [str(i), new_labels[i]['Name']]
            writer.writerow(towrite)

    return

clean_data('data/images')
