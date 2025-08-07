from data_loader import load_data, prepare_data, print_data_info
from quantization import AdaptiveQuantizationKernel
from feature_extraction import FeatureReductionModule
from model import DualDomainFusionTransformer
from trainer import train_model
from svm import train_svm_with_hyperparameter_tuning, evaluate_svm
import torch
import torch.utils.data
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
import torch.nn as nn

# Load data
file_path = 'D:/ddst/industrial.csv'
data = load_data(file_path)
X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = prepare_data(data)
print_data_info(X_train_raw, X_val_raw, X_test_raw)

# Initialize and fit the quantization kernel
time_kernel = AdaptiveQuantizationKernel(M=100, alpha=0.3)
time_kernel.fit(X_train_raw)
X_train_time_raw = time_kernel.transform(X_train_raw)
X_val_time_raw = time_kernel.transform(X_val_raw)
X_test_time_raw = time_kernel.transform(X_test_raw)

# Standardize time domain features
scaler_time = StandardScaler()
X_train_time = scaler_time.fit_transform(X_train_time_raw)
X_val_time = scaler_time.transform(X_val_time_raw)
X_test_time = scaler_time.transform(X_test_time_raw)

# Standardize frequency domain features
def process_freq_domain(X):
    return np.abs(np.fft.fft(X, axis=1))  # Frequency domain features

X_train_freq_raw = process_freq_domain(X_train_raw)
X_val_freq_raw = process_freq_domain(X_val_raw)
X_test_freq_raw = process_freq_domain(X_test_raw)

scaler_freq = StandardScaler()
X_train_freq = scaler_freq.fit_transform(X_train_freq_raw)
X_val_freq = scaler_freq.transform(X_val_freq_raw)
X_test_freq = scaler_freq.transform(X_test_freq_raw)

# Combine time and frequency features
X_train_combined = np.concatenate((X_train_time, X_train_freq), axis=1)
X_val_combined = np.concatenate((X_val_time, X_val_freq), axis=1)
X_test_combined = np.concatenate((X_test_time, X_test_freq), axis=1)

# Prepare dataset for PyTorch DataLoader
class DNADataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = DNADataset(X_train_combined, y_train)
val_dataset = DNADataset(X_val_combined, y_val)
test_dataset = DNADataset(X_test_combined, y_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize model
time_dim = X_train_time.shape[1]
freq_dim = X_train_freq.shape[1]

model = DualDomainFusionTransformer(time_dim, freq_dim, d_model=128).to(device)

# Training settings
criterion = nn.BCELoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.0001)

# Train the model
trained_model = train_model(model, train_loader, val_loader, criterion, optimizer)

# Use SVM for evaluation
svm_model, svm_scaler = train_svm_with_hyperparameter_tuning(X_train_time, y_train)
precision, f1 = evaluate_svm(svm_model, svm_scaler, X_test_time, y_test)

print(f"SVM Precision: {precision:.4f}, F1 Score: {f1:.4f}")
