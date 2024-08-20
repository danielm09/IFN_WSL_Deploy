import torch

def evaluate(model, dataloader, criterion, num_classes):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            # Define the mask for loss computation
            mask = labels!=255
            mask_expanded = mask.unsqueeze(1).expand(-1, num_classes, -1, -1)
            outputs = model(inputs)
            outputs_selected = outputs[mask_expanded].view(-1, num_classes)
            labels_selected = labels[mask]
            loss = criterion(outputs_selected, labels_selected)
            total_loss += loss.item()

            predicted_selected = torch.argmax(outputs_selected, dim=1) #no need to have softmax applied earlier
            total += labels_selected.size(0)
            correct += (predicted_selected == labels_selected).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

def outputPredictions(model, dataloader, criterion):
    model.eval()  # Set the model to evaluation mode
    target = []
    preds = []
    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)

            preds.append(predicted)
            target.append(labels)

    return target, preds
