import torch
import torch.optim as optim
import torch.nn as nn

def train_model(model, train_loader, val_loader, criterion, optimizer, n_epochs=200):
    best_model_state = None
    best_val_accuracy = 0.0

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            (time_input, freq_input), y = batch
            time_input, freq_input, y = time_input.to(device), freq_input.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(time_input, freq_input)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        val_loss = validate_model(model, val_loader, criterion)
        val_accuracy = calculate_val_accuracy(model, val_loader)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()

    model.load_state_dict(best_model_state)
    return model

def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            (time_input, freq_input), y = batch
            time_input, freq_input, y = time_input.to(device), freq_input.to(device), y.to(device)
            output = model(time_input, freq_input)
            loss = criterion(output, y)
            val_loss += loss.item()
    return val_loss / len(val_loader)

def calculate_val_accuracy(model, val_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            (time_input, freq_input), y = batch
            time_input, freq_input, y = time_input.to(device), freq_input.to(device), y.to(device)
            output = model(time_input, freq_input)
            predicted = (output >= 0.5).float()
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return correct / total
