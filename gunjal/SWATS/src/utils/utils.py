import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt
import pandas as pd

from utils.swats import Swats

def data_loader(batch_size, num_workers, shuffle):
    # train and test transformers 
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),
                             (0.2023,0.1994,0.2010))
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),
                                (0.2023,0.1994,0.2010))
    ])

    # train and test datasets
    train_dataset = datasets.CIFAR10(root='./data',
                                     train=True,
                                     download=True,
                                     transform=train_transforms)
    test_dataset = datasets.CIFAR10(root='./data', 
                                    train=False,
                                    download=True,
                                    transform=test_transforms)
    
    # split train dataset into train and validation
    train_dataset, val_dataset = random_split(train_dataset, [int(len(train_dataset)*0.8), int(len(train_dataset)*0.2)])

    print(f"Number of training examples: {len(train_dataset)}")
    print(f"Number of validation examples: {len(val_dataset)}")
    print(f"Number of testing examples: {len(test_dataset)}")

    # train, validation and test dataloaders
    train_loader = DataLoader(train_dataset,
                                batch_size = batch_size,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                pin_memory=True)
    
    val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                pin_memory=True)
    
    test_loader = DataLoader(test_dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                pin_memory=True)
       
    return train_loader, val_loader, test_loader

def select_optimizer(optimizer, model, learning_rate):
    """
    Create and return an optimizer based on the specified type.

    Args:
        optimizer_type (str): The type of optimizer to use ('adam' or 'sgd').
        learning_rate (float): The learning rate for the optimizer.

    Returns:
        torch.optim: The selected optimizer.
    """
    if optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate, 
                                     betas=(0.9, 0.999), eps=1e-08,
                                     weight_decay=0, amsgrad=False)
    elif optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer == "SWATS":
        optimizer = Swats(model.parameters())
    else:
        raise ValueError("Invalid optimizer")
    
    return optimizer

def plot_loss(train_loss, val_loss):
    df = pd.DataFrame({'train_loss': train_loss, 'val_loss': val_loss})
    df.plot(linewidth=4, alpha = 0.7, figsize=(15, 7), label = 'Loss')

    plt.xlim([0,10])

    plt.title('Training vs Validation Loss per Epoch', fontsize=22)
    plt.grid(axis='y', alpha=0.5)
    plt.yticks(fontsize=12, alpha=.7)
    plt.xticks(fontsize=12, alpha=.7)
    plt.xlabel('Epoch', fontsize=18, alpha=.7)
    plt.ylabel('Loss Value', fontsize=18, alpha=.7)

    plt.gca().spines["top"].set_alpha(0.0)
    plt.gca().spines["bottom"].set_alpha(0.3)
    plt.gca().spines["right"].set_alpha(0.0)
    plt.gca().spines["left"].set_alpha(0.3)

    plt.legend(loc = 'upper right', fontsize=16)
    plt.show()

def plot_accuracy(train_acc, val_acc):
    df = pd.DataFrame({'Training Accuracy':train_acc, 'Validation Accuracy':val_acc})
    df.plot(linewidth=4, alpha=0.7, figsize=(15,7), label='Loss')
    plt.xlim([0,10])
    # plt.ylim(-20,100)
    plt.title('Training vs Validation Accuracy Per Epoch', fontsize=22)
    plt.grid(axis='y', alpha=.5)
    plt.yticks(fontsize=12, alpha=.7)
    plt.xticks(fontsize=12, alpha=.7)
    plt.xlabel('Epoch', fontsize=18, alpha=.7)
    plt.ylabel('Accuracy Percentage', fontsize=18, alpha=.7)
    # Lighten borders
    plt.gca().spines["top"].set_alpha(.0)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(.0)
    plt.gca().spines["left"].set_alpha(.3)

    plt.legend(loc='upper right')
    plt.show()
