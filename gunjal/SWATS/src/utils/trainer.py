import torch
import wandb
from tqdm import tqdm

def trainer(model, config, train_loader, val_loader, criterion, optimizer, device):
    best_accuracy = 0.0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    # ------------------------------ TRAINING ------------------------------#
    for epoch in range(config.epochs):
        model.train()
        total_train_loss = 0.0
        correct_train_predictions = 0

        train_loader_with_progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.epochs}')

        for batch_index, (images, targets) in enumerate(train_loader_with_progress):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train_predictions += torch.sum(preds == targets).item()

            train_loader_with_progress.set_postfix({'Loss': loss.item()})

        average_train_loss = total_train_loss / len(train_loader)
        train_accuracy = correct_train_predictions / len(train_loader.dataset) * 100
        train_losses.append(average_train_loss)
        train_accuracies.append(train_accuracy)

        print(f'Epoch: {epoch + 1} | Training Loss: {average_train_loss} | Training Accuracy: {train_accuracy}%')

        # ------------------------------ VALIDATION ------------------------------#
        model.eval()
        total_val_loss = 0.0
        correct_val_predictions = 0

        val_loader_with_progress = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{config.epochs}')

        with torch.no_grad():
            for images, targets in val_loader_with_progress:
                images = images.to(device)
                targets = targets.to(device)

                outputs = model(images)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_val_predictions += torch.sum(preds == targets).item()

                val_loader_with_progress.set_postfix({'Loss': loss.item()})

        average_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_val_predictions / len(val_loader.dataset) * 100
        val_losses.append(average_val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch: {epoch + 1} | Validation Loss: {average_val_loss} | Validation Accuracy: {val_accuracy}%')

        # Early stopping and model saving
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            model_filename = f"Accuracy_{best_accuracy}.ckpt"
            config.model_path = config.model_path + model_filename
            print('Saving the model...')
            torch.save(model.state_dict(), config.model_path)

    return train_losses, val_losses, train_accuracies, val_accuracies
