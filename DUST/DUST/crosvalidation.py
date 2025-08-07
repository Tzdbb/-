import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import os
from sklearn.metrics import roc_auc_score, confusion_matrix

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 读取数据
file_path = r'D:\ddst\industrial.csv'  # 更新文件路径
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit()

try:
    # 加载数据
    data = pd.read_csv(file_path, encoding='utf-8', sep=',')
    print("File loaded successfully.")
    print(data.head())
except Exception as e:
    print("Failed to load file.")
    print(e)
    exit()

# 检查数据格式
if data.shape[1] < 2:
    print("The data file does not have enough columns for CIR and labels.")
    exit()

# 标签列：将'NLOS'转换为1，其他转换为0
y = data.iloc[:, 0].apply(lambda x: 1 if x.strip().upper() == "NLOS" else 0).values  # NLOS为1，LOS为0

# 处理CIR数据，将字符串转换为复数
def parse_complex(value):
    try:
        return complex(value.strip("()"))  # 去掉括号并转换为 complex 类型
    except:
        return 0 + 0j  # 解析失败时返回默认值 0 + 0j

X_complex = data.iloc[:, 1:].applymap(parse_complex).values

# 数据划分：80%训练集，20%测试集
X_train_temp, X_test_raw, y_train_temp, y_test = train_test_split(X_complex, y, test_size=0.2, random_state=42)

# 训练集与验证集分割：训练集80%，验证集20%
X_train_raw, X_val_raw, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.2, random_state=42)

print(f"训练集大小: {len(X_train_raw)}")
print(f"验证集大小: {len(X_val_raw)}")
print(f"测试集大小: {len(X_test_raw)}")

# 频域处理
def process_freq_domain(X):
    return np.abs(np.fft.fft(X, axis=1))  # 频域特征的振幅部分

X_train_freq_raw = process_freq_domain(X_train_raw)
X_val_freq_raw = process_freq_domain(X_val_raw)
X_test_freq_raw = process_freq_domain(X_test_raw)

# 标准化
scaler_freq = StandardScaler()
X_train_freq = scaler_freq.fit_transform(X_train_freq_raw)
X_val_freq = scaler_freq.transform(X_val_freq_raw)
X_test_freq = scaler_freq.transform(X_test_freq_raw)

# 特征合并
X_train_combined = np.concatenate((X_train_freq, X_train_raw), axis=1)
X_val_combined = np.concatenate((X_val_freq, X_val_raw), axis=1)
X_test_combined = np.concatenate((X_test_freq, X_test_raw), axis=1)

# 训练SVM模型并进行五折交叉验证
def train_svm_with_cross_validation(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 使用StratifiedKFold进行交叉验证
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    svm = SVC(kernel='rbf', probability=True)

    # 超参数调优：通过交叉验证调整SVM的超参数
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10]
    }

    grid_search = GridSearchCV(svm, param_grid, cv=kf, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train.ravel())

    return grid_search.best_estimator_, scaler

# 评估SVM模型
def evaluate_svm(svm, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    predictions = svm.predict(X_test_scaled)

    precision = precision_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, svm.predict_proba(X_test_scaled)[:, 1])

    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return accuracy, precision, recall, f1, specificity, sensitivity, auc

# 进行SVM训练和评估
best_svm_model, scaler = train_svm_with_cross_validation(X_train_combined, y_train)

# 评估SVM模型
test_accuracy, test_precision, test_recall, test_f1, test_specificity, test_sensitivity, test_auc = evaluate_svm(best_svm_model, scaler, X_test_combined, y_test)

# 打印测试结果
print("SVM Test Results:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1 Score: {test_f1:.4f}")
print(f"Specificity: {test_specificity:.4f}")
print(f"Sensitivity: {test_sensitivity:.4f}")
print(f"AUC: {test_auc:.4f}")
