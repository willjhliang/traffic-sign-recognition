
from sklearn.decomposition import PCA
import numpy as np
from matplotlib import pyplot as plt


def run_pca(X_train_flattened, X_test_flattened):
    covar_matrix = PCA(n_components=min(X_train_flattened.shape[0], 32*32))
    covar_matrix.fit(X_train_flattened)
    var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)
    plt.plot(var[:300]);

    pca = PCA(n_components=150)
    pca.fit(X_train_flattened)
    X_train_pca = pca.transform(X_train_flattened)
    X_test_pca = pca.transform(X_test_flattened)
    return X_train_pca, X_test_pca, pca

def visualize_pca(pca):
    fig, axes = plt.subplots(3, 8, figsize=(9, 3.5),
        subplot_kw={'xticks':[], 'yticks':[]},
        gridspec_kw=dict(hspace=0.1, wspace=0.1)
    )
    
    for i, ax in enumerate(axes.flat):
        img = pca.components_[i].reshape(3, 32, 32)
        min_val, max_val = np.min(img), np.max(img)
        img = (img - min_val) / (max_val - min_val)
        img = np.swapaxes(img, 0, -1)
        ax.imshow(img)
    plt.show()

def visualize_pca_per_channel(X_train):
    def pca_on_channel(channel):
        X_train_channel = np.array([i.flatten() for i in X_train[:, channel, :, :]])
        ret = PCA(n_components=150)
        ret.fit(X_train_channel)
        return ret

    fig, axes = plt.subplots(3, 8, figsize=(9, 4),
        subplot_kw={'xticks':[], 'yticks':[]},
        gridspec_kw=dict(hspace=0.1, wspace=0.1)
    )
    for i, ax in enumerate(axes.flat):
        ax.imshow(pca_on_channel(0).components_[i].reshape(32, 32))
    
    fig, axes = plt.subplots(3, 8, figsize=(9, 4),
        subplot_kw={'xticks':[], 'yticks':[]},
        gridspec_kw=dict(hspace=0.1, wspace=0.1)
    )
    for i, ax in enumerate(axes.flat):
        ax.imshow(pca_on_channel(1).components_[i].reshape(32, 32))
        
    fig, axes = plt.subplots(3, 8, figsize=(9, 4),
        subplot_kw={'xticks':[], 'yticks':[]},
        gridspec_kw=dict(hspace=0.1, wspace=0.1)
    )
    for i, ax in enumerate(axes.flat):
        ax.imshow(pca_on_channel(2).components_[i].reshape(32, 32))