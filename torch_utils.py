
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.utils import data
from torch import Tensor
from torchmetrics import Accuracy, Precision, Recall, F1Score

from constants import K
from data import split_validation


def load_torch_data(X_train, y_train, X_test, y_test):
    """Converts numpy data to torch dataloaders."""
    X_train, X_val, y_train, y_val = split_validation(X_train, y_train)
    train_set = data.TensorDataset(Tensor(X_train), Tensor(y_train))
    val_set = data.TensorDataset(Tensor(X_val), Tensor(y_val))
    test_set = data.TensorDataset(Tensor(X_test), Tensor(y_test))
    train_loader = data.DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = data.DataLoader(val_set, batch_size=32, shuffle=True)
    test_loader = data.DataLoader(test_set, batch_size=32, shuffle=True)
    return train_loader, val_loader, test_loader
    

def train_model(model, train_loader, val_loader, epochs, lr):
    """Trains model and plots loss and accuracies."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr) 

    losses = []
    train_accuracy, val_accuracy = [], []
    train_precision, val_precision = [], []
    train_recall, val_recall = [], []
    train_f1, val_f1 = [], []
    for epoch in range(epochs):
        running_loss = 0
        for itr, (image, label) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
            optimizer.zero_grad()
            y_predicted = model(image)
            label = label.long()

            loss = criterion(y_predicted, label)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
    
        losses.append(running_loss)
        _, (accuracy, precision, recall, f1) = evaluate_model(model, train_loader)
        train_accuracy.append(accuracy)
        train_precision.append(precision)
        train_recall.append(recall)
        train_f1.append(f1)
        _, (accuracy, precision, recall, f1) = evaluate_model(model, val_loader)
        val_accuracy.append(accuracy)
        val_precision.append(precision)
        val_recall.append(recall)
        val_f1.append(f1)
        print(f'Epoch: {epoch+1:03}, Loss: {running_loss:9.4f}, Train F1: {train_f1[-1]:.4f}, Validation F1: {val_f1[-1]:.4f}')

    fig, axs = plt.subplots(1, 5, figsize=(20, 2.5))
    axs[0].plot(losses)
    axs[1].plot(list(range(epochs)), train_accuracy, val_accuracy);
    axs[2].plot(list(range(epochs)), train_precision, val_precision);
    axs[3].plot(list(range(epochs)), train_recall, val_recall);
    axs[4].plot(list(range(epochs)), train_f1, val_f1);

    return model


def evaluate_model(model, dataloader):
    """Evaluates model and returns predictions and metrics."""
    accuracy = Accuracy(task='multiclass', num_classes=K)
    precision = Precision(task='multiclass', num_classes=K)
    recall = Recall(task='multiclass', num_classes=K)
    f1 = F1Score(task='multiclass', num_classes=K)

    pred = torch.Tensor([])
    with torch.no_grad():
        for _, (images, labels) in enumerate(dataloader):
            outputs = model(images)
            _, batch_pred = torch.max(outputs.data, 1)
            pred = torch.cat((pred, batch_pred))
            accuracy(batch_pred, labels)
            precision(batch_pred, labels)
            recall(batch_pred, labels)
            f1(batch_pred, labels)

    return pred.tolist(), (accuracy.compute(), precision.compute(), recall.compute(), f1.compute())