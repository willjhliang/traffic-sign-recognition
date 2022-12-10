
import numpy as np
from matplotlib import pyplot as plt
from heapq import heappush, heappop
from PIL import Image
import torch
from sklearn.manifold import TSNE

from data import load_data, consolidate_data


def visualize_top_activations(model, X_train):
    """Visualizes images that have highest activations in certain neurons after last convolutional layer."""
    indices = [0, 8, 24]
    activations = [[] for _ in range(len(indices))]

    for i in range(X_train.shape[0]):
        img = X_train[i]
        x = torch.from_numpy(np.expand_dims(X_train[i].astype(np.float32), 0))
        x = model.relu(model.batch_norm_1(model.conv_1(x)))
        x = model.relu(model.batch_norm_2(model.conv_2(x)))
        x = model.max_pool2d(x)
        x = model.relu(model.batch_norm_3(model.conv_3(x)))
        x = model.relu(model.batch_norm_4(model.conv_4(x)))
        x = model.dropout_1(x)
        x = model.max_pool2d(x)
        x = model.relu(model.batch_norm_5(model.conv_5(x)))
        x = model.relu(model.batch_norm_6(model.conv_6(x)))
        x = model.dropout_2(x)
        x = model.max_pool2d(x)
        for j, idx in enumerate(indices):
            heappush(activations[j], (x[0, idx, 0, 0].item(), i))
            if len(activations[j]) > 8:
                heappop(activations[j])
    
    print(f'--- Top activations for final convolutional layer index {indices} ---')
    _, axs = plt.subplots(len(indices), 8, figsize=(8, 1.5*len(indices)))
    for j in range(len(indices)):
        for i in range(8):
            axs[j, i].axis('off')
            img = X_train[heappop(activations[j])[1]]
            axs[j, i].imshow(np.swapaxes(img, 0, -1))


def visualize_tsne_similarity(model, X_train):
    """Visualizes image similarity measured by activations after last convolutional layer."""
    features = []
    with torch.no_grad():
        for i in range(200):
            img = X_train[i]
            x = torch.from_numpy(np.expand_dims(X_train[i].astype(np.float32), 0))
            x = model.relu(model.batch_norm_1(model.conv_1(x)))
            x = model.relu(model.batch_norm_2(model.conv_2(x)))
            x = model.max_pool2d(x)
            x = model.relu(model.batch_norm_3(model.conv_3(x)))
            x = model.relu(model.batch_norm_4(model.conv_4(x)))
            x = model.dropout_1(x)
            x = model.max_pool2d(x)
            x = model.relu(model.batch_norm_5(model.conv_5(x)))
            x = model.relu(model.batch_norm_6(model.conv_6(x)))
            x = model.dropout_2(x)
            x = model.max_pool2d(x)
            features.append(x.numpy().flatten())
    
    tsne = TSNE(n_components=2, learning_rate=150, perplexity=30, angle=0.2, verbose=2).fit_transform(features)
    tx, ty = tsne[:,0], tsne[:,1]
    tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

    width = 2000
    height = 2000
    max_dim = 100

    full_image = Image.new('RGBA', (width, height))
    for img, x, y in zip(X_train, tx, ty):
        tile = Image.fromarray((np.swapaxes(img, 0, -1) * 255).astype(np.uint8)).resize((224, 224))
        rs = max(1, tile.width/max_dim, tile.height/max_dim)
        tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
        full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))

    print('--- Image similarity measured by final convolutional activations ---')
    plt.figure(figsize = (16,12))
    plt.imshow(full_image)


def adversarial_attack(model, labels):
    """Tests model against adversarial images, visualizing mispredictions."""
    adversarial_data = load_data('data/adversarial_images')
    X_adv, _, y_adv = consolidate_data(adversarial_data)
    failed = []
    for i in range(X_adv.shape[0]):
        out = model(torch.from_numpy(np.expand_dims(X_adv[i], 0).astype(np.float32)))
        _, pred = torch.max(out.data, 1)
        if pred != y_adv[i]:
            failed.append((i, pred.item()))
    failed = failed[:8]

    print('--- Failed adversarial examples ---')
    _, axs = plt.subplots(1, len(failed), figsize=(len(failed) * 2.5, 1.5))
    for i, (img_idx, pred) in enumerate(failed):
        axs[i].axis('off')
        img = X_adv[img_idx]
        axs[i].imshow(np.swapaxes(img, 0, -1))
        axs[i].title.set_text(labels['Name'][pred])