from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, f1_score, recall_score


def train_svm_with_hyperparameter_tuning(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10]}
    svm = SVC(kernel='rbf', probability=True)
    grid_search = GridSearchCV(svm, param_grid, scoring='f1', cv=5)
    grid_search.fit(X_train_scaled, y_train.ravel())

    return grid_search.best_estimator_, scaler


def evaluate_svm(svm_model, svm_scaler, X_test, y_test):
    X_test_scaled = svm_scaler.transform(X_test)
    predictions = svm_model.predict(X_test_scaled)
    precision = precision_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    return precision, f1, recall
