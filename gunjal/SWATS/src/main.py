import torch
import wandb
import argparse

from utils.utils import data_loader, select_optimizer
from utils.model import ResNet, BasicBlock
from utils.trainer import trainer
from utils.test import test_model

# WandB login
wandb.login(key=" ")

def train(config, model, train_loader, val_loader, device):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = select_optimizer("SGD", model, config.learning_rate)
    
    wandb.init(project="SWATS", entity="", config={
                                "learning_rate": config.learning_rate,
                                "momentum": config.momentum,
                                "batch_size": config.batch_size,
                                "epochs": config.epochs
                            })
    wandb.watch(model, criterion, log="all", log_freq=10)
    
    train_loss, val_loss, train_acc, val_acc = trainer(model, config, train_loader, val_loader, criterion, optimizer, device)
    
    print(f'Accuracy of the network on the train dataset: {max(train_acc)}')
    print(f'Accuracy of the network on the validation dataset: {max(val_acc)}')

    wandb.log({"Train Loss": train_loss, "Validation Loss": val_loss})
    wandb.log({"Train Accuracy": max(train_acc), "Validation Accuracy": max(val_acc)})
    wandb.log({"Loss": wandb.plot.line_series(list(range(config.epochs)), [train_loss, val_loss], ["Train Loss", "Validation Loss"], title="Loss")})

    wandb.finish()

def test(config, model, test_loader, best_model, device):
    test_accuracy, correct_predictions, wrong_predictions, all_targets, all_preds = test_model(model,
                                                            test_loader, best_model, device)

    print(f'Accuracy of the network on the test dataset: {round(test_accuracy)}')
    print(f'Correct predictions: {len(correct_predictions)}')
    print(f'Wrong predictions: {len(wrong_predictions)}')
    print(f'All predictions: {len(all_preds)}')
    print(f'All targets: {len(all_targets)}')

def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = data_loader(config.batch_size, config.num_workers, config.shuffle)

    model = ResNet(BasicBlock, [3, 4, 6, 3])
    model.to(device)

    if config.mode == "train":
        train(config, model, train_loader, val_loader, device)
    elif config.mode == "test":
        best_model = " "
        test(config, model, test_loader, best_model, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or test ResNet model.")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], required=True, help="Mode: train or test")
    parser.add_argument("--model_path", type=str, default="./SWATS/", help="Path to save/load the model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data loader workers")
    parser.add_argument("--shuffle", type=bool, default=True, help="Shuffle data during training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    

    args = parser.parse_args()

    main(args)
