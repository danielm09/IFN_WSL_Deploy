import torch

def evaluate(model, dataloader, criterion, num_classes):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)
            mask = labels!=20 #20 = new value for 255
            labels_selected = labels[mask]
            predicted_selected = predicted[mask]
            correct += (predicted_selected == labels_selected).sum().item()  # Sum the correct predictions
            total += labels_selected.size(0)

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
