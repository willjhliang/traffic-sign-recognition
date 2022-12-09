
import random
import torch
import numpy as np


K = 36                  # Number of classes
S = 32                  # Size of image, dimension is (s, s, 3)
class_size = 320        # Number of images per class
validation_ratio = 0.1  # Proportion of training data to set aside for validation

random_seed = 19104     # Seed all random operations to ensure reproducibility
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)