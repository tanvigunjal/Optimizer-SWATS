import torch

def test_model(model, test_loader, BEST_MODEL, device):
    model.load_state_dict(torch.load(BEST_MODEL))
    model.eval()

    correct_predictions = []
    wrong_predictions = []
    all_targets = []
    all_preds = []
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct_mask = preds == targets
            total_correct += torch.sum(correct_mask).item()
            total_samples += targets.size(0)

            correct_indices = torch.where(correct_mask)
            wrong_indices = torch.where(~correct_mask)

            correct_predictions.extend(
                zip(images[correct_indices], targets[correct_indices], preds[correct_indices])
            )
            wrong_predictions.extend(
                zip(images[wrong_indices], targets[wrong_indices], preds[wrong_indices])
            )

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = (total_correct / total_samples) * 100

    return accuracy, correct_predictions, wrong_predictions, all_targets, all_preds