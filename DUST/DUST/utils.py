import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.utils.data

def cir_to_frequency_domain(cir_sequences):
    """
    Converts CIR time-series data to frequency domain.
    """
    freq_sequences = np.fft.fft(cir_sequences, axis=1)
    return np.abs(freq_sequences)


import torch
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

def calculate_metrics(predictions, labels):
    """
    Calculates accuracy, sensitivity, and AUC-ROC.
    """
    binary_predictions = (predictions >= 0.5).astype(int).flatten()
    labels = labels.astype(int).flatten()

    tn, fp, fn, tp = confusion_matrix(labels, binary_predictions).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    auc_roc = roc_auc_score(labels, predictions) if len(set(labels)) > 1 else 0.0

    return accuracy, sensitivity, auc_roc

def extract_features(model, iterator):
    """Extract features using the trained model"""
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for batch in iterator:
            (time_input, freq_input), y = batch
            time_input, freq_input = time_input.to(device), freq_input.to(device)

            output = model(time_input, freq_input)
            probabilities = torch.sigmoid(output).cpu().numpy()
            features.extend(probabilities)
            labels.extend(y.cpu().numpy())

    return np.array(features), np.array(labels)

def calculate_loss(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in iterator:
            (time_input, freq_input), y = batch
            time_input, freq_input, y = time_input.to(device), freq_input.to(device), y.to(device)

            output = model(time_input, freq_input)
            loss = criterion(output, y)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)
