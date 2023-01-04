import os
import pickle

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from model import MyAwesomeModel
from torch import optim
from torch.utils.data import DataLoader, Dataset



class dataset(Dataset):
    def __init__(self, images, labels):
        self.data = images.view(-1, 1, 28, 28)
        self.labels = labels

    def __getitem__(self, item):
        return self.data[item].float(), self.labels[item]

    def __len__(self):
        return len(self.data)


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
@click.option("--epoch", default=30, help='learning rate to use for training')
@click.option("--batch_size", default=16, help='learning rate to use for training')
def train(lr, epoch, batch_size):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()

    with open("data/processed/corruptmnist/corruptmnist_train.npz", "rb") as fb:
        train_images, train_labels = pickle.load(fb)

    train_dataset = dataset(train_images, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = epoch
    if not os.path.exists(f'models/{model.name}/'):
        os.makedirs(f'models/{model.name}/')

    if not os.path.exists(f"reports/figures/{model.name}/"):
        os.makedirs(f"reports/figures/{model.name}/")

    train_losses = []
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_loader:

            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        else:
            if e == 0:
                torch.save(model.state_dict(), f'models/{model.name}/checkpoint.pth')
                print(f"Model saved at epoch: {e} with running loss: {running_loss}")
            else:
                if running_loss < min(train_losses):
                    torch.save(model.state_dict(), f'models/{model.name}/checkpoint.pth')
                    print(f"Model saved at epoch: {e} with running loss: {running_loss}")

            train_losses += [running_loss/len(train_loader)]

            plt.plot(np.arange(0, e+1), train_losses, color='royalblue')
            plt.title(f"Training loss curve at epoch: {e}")
            plt.grid()
            plt.savefig(f"reports/figures/{model.name}/{model.name}"+"training_curve")

            ps = torch.exp(model(images))
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            print(f'Epoch: {e}, Loss: {running_loss/len(train_loader)}, Accuracy: {accuracy.item() * 100}%\n')


if __name__ == "__main__":
    train()
