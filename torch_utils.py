
import torch
from torch import nn
from torch import optim
from torch.utils import data
from torch import Tensor
from matplotlib import pyplot as plt
from tqdm import tqdm

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

    train_loss = []
    train_acc = []
    val_acc = []
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
    
        train_loss.append(running_loss)
        train_acc.append(evaluate_model(model, train_loader)[-1])
        val_acc.append(evaluate_model(model, val_loader)[-1])
        print(f'Epoch: {epoch+1:03}, Loss: {running_loss:9.4f}, Train Accuracy: {train_acc[-1]:.4f}, Validation Accuracy: {val_acc[-1]:.4f}')

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(train_loss)
    axs[1].plot(list(range(epochs)), train_acc, val_acc);

    return model


def evaluate_model(model, dataloader):
    """Evaluates model and returns predictions and accuracy."""
    y_pred = []
    y_pred = torch.Tensor()
    with torch.no_grad():
        for _, (image, _) in enumerate(dataloader):
            outputs = model(image)
            _, predicted = torch.max(outputs.data, 1)
            y_pred = torch.cat((y_pred, predicted))

    return y_pred.tolist()