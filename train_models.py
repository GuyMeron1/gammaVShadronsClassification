# Imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import warnings
from sklearn.exceptions import ConvergenceWarning

# Load Dataset
cols = ["fLength", "fWidth", "fSize", "fConc", "fConcl", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv("../ML - MAGICGammaTelescope/magic04.data", names=cols)
df["class"] = (df["class"] == "g").astype(int)

# Split Dataset
train, valid, test = np.split(df.sample(frac=1, random_state=42),[int(0.6*len(df)), int(0.8*len(df))])

# Scaling & Optional Oversampling
def scale_dataset(dataframe, scaler=None, oversample=False, fit=True):
    X = dataframe[dataframe.columns[:-1]].values
    Y = dataframe[dataframe.columns[-1]].values
    if fit:
        scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    if oversample:
        ros = RandomOverSampler()
        X, Y = ros.fit_resample(X, Y)
    return X, Y, scaler

X_train, Y_train, scaler = scale_dataset(train, oversample=True, fit=True)
X_valid, Y_valid, _ = scale_dataset(valid, scaler=scaler, fit=False)
X_test, Y_test, _ = scale_dataset(test, scaler=scaler, fit=False)

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
# Save train data for visualization file
with open("train_data.pkl", "wb") as f:
    pickle.dump((X_train, Y_train), f)
# Save test data for visualization file
with open("test_data.pkl", "wb") as f:
    pickle.dump((X_test, Y_test), f)

# Train Models
def train_knn():
    param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 15, 21, 31, 51, 71, 91], 'p': [1, 2]}
    metrics = {}

    # Try all combinations on the validation set
    for k in param_grid['n_neighbors']:
        for p in param_grid['p']:
            knn = KNeighborsClassifier(n_neighbors=k, p=p)
            knn.fit(X_train, Y_train)
            Y_val_pred = knn.predict(X_valid)
            metrics[(k, p)] = {
                "Accuracy": accuracy_score(Y_valid, Y_val_pred),
                "F1": f1_score(Y_valid, Y_val_pred, average="macro")
            }

    # Save metrics for visualization file
    with open("knn_metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)

    # Pick best model according to validation F1
    best_params = max(metrics, key=lambda x: metrics[x]['F1'])
    best_knn = KNeighborsClassifier(n_neighbors=best_params[0], p=best_params[1])

    # Retrains the best KNN model using both the training and validation data combined,
    # so it learns from all available labeled examples before final evaluation or saving.
    best_knn.fit(
        np.vstack([X_train, X_valid]),
        np.hstack([Y_train, Y_valid])
    )

    # Save the best model
    with open("knn_model.pkl", "wb") as f:
        pickle.dump(best_knn, f)

    print(f"KNN - Best params (from validation): (k={best_params[0]}, p={best_params[1]})")
    return best_knn

def train_nb():
    param_grid = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]}
    metrics = {}

    # Try all combinations on the validation set
    for vs in param_grid['var_smoothing']:
        nb = GaussianNB(var_smoothing=vs)
        nb.fit(X_train, Y_train)
        Y_val_pred = nb.predict(X_valid)
        metrics[vs] = {
            "Accuracy": accuracy_score(Y_valid, Y_val_pred),
            "F1": f1_score(Y_valid, Y_val_pred, average='macro')
        }

    # Pick best model according to validation F1
    best_vs = max(metrics, key=lambda x: metrics[x]['F1'])
    best_nb = GaussianNB(var_smoothing=best_vs)

    # Retrains the best Naive Bayes model using both the training and validation data combined,
    # so it can learn from all available labeled examples before final evaluation or saving.
    best_nb.fit(
        np.vstack([X_train, X_valid]),
        np.hstack([Y_train, Y_valid])
    )

    # Save the best model
    with open("naive_bayes_model.pkl", "wb") as f:
        pickle.dump(best_nb, f)

    print(f"Naive Bayes - Best params (from validation): ({best_vs})")
    return best_nb

def train_dt():
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 3, 4, 5, 6],
        'min_samples_split': [4, 5, 6],
        'min_samples_leaf': [2, 3, 4]
    }
    metrics = {}

    # Try all combinations on the validation set
    for criterion in param_grid['criterion']:
        for max_depth in param_grid['max_depth']:
            for min_split in param_grid['min_samples_split']:
                for min_leaf in param_grid['min_samples_leaf']:
                    dt = DecisionTreeClassifier(
                        criterion=criterion,
                        max_depth=max_depth,
                        min_samples_split=min_split,
                        min_samples_leaf=min_leaf
                    )
                    dt.fit(X_train, Y_train)
                    Y_val_pred = dt.predict(X_valid)
                    metrics[(criterion, max_depth, min_split, min_leaf)] = {
                        "Accuracy": accuracy_score(Y_valid, Y_val_pred),
                        "F1": f1_score(Y_valid, Y_val_pred, average='macro')
                    }

    # Pick best model according to validation F1
    best_params = max(metrics, key=lambda x: metrics[x]['F1'])
    best_dt = DecisionTreeClassifier(
        criterion=best_params[0],
        max_depth=best_params[1],
        min_samples_split=best_params[2],
        min_samples_leaf=best_params[3]
    )

    # Retrains the best decision tree model using both the training and validation data combined,
    # so it can learn from all available labeled examples before final evaluation or saving.
    best_dt.fit(
        np.vstack([X_train, X_valid]),
        np.hstack([Y_train, Y_valid])
    )

    # Save the best model
    with open("decision_tree_model.pkl", "wb") as f:
        pickle.dump(best_dt, f)

    print(f"Decision Tree - Best params (from validation): {best_params}")
    return best_dt

def train_svm():
    param_grid = {'C': [0.1, 1, 10],
                  'gamma': [0.1, 0.01],
                  'kernel': ['linear', 'rbf']}
    metrics = {}

    # Try all combinations on the validation set
    for C in param_grid['C']:
        for gamma in param_grid['gamma']:
            for kernel in param_grid['kernel']:
                svm = SVC(C=C, gamma=gamma, kernel=kernel, probability=True)
                svm.fit(X_train, Y_train)
                Y_val_pred = svm.predict(X_valid)
                metrics[(C, gamma, kernel)] = {
                    "Accuracy": accuracy_score(Y_valid, Y_val_pred),
                    "F1": f1_score(Y_valid, Y_val_pred, average='macro')
                }

    # Pick best model according to validation F1
    best_params = max(metrics, key=lambda x: metrics[x]['F1'])
    best_svm = SVC(C=best_params[0], gamma=best_params[1], kernel=best_params[2], probability=True)

    # Retrains the svm model using both the training and validation data combined,
    # so it can learn from all available labeled examples before final evaluation or saving.
    best_svm.fit(
        np.vstack([X_train, X_valid]),
        np.hstack([Y_train, Y_valid])
    )

    # Save the best model
    with open("svm_model.pkl", "wb") as f:
        pickle.dump(best_svm, f)

    print(f"SVM best params (from validation): (C={best_params[0]}, gamma={best_params[1]}, kernel={best_params[2]})")
    return best_svm

def train_nn():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    param_grid = {
        'hidden_layer_sizes': [(50,), (100,)],
        'activation': ['relu'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.001],
        'learning_rate': ['constant', 'adaptive']
    }
    metrics = {}

    # Try all combinations on the validation set
    for hidden_layer in param_grid['hidden_layer_sizes']:
        for alpha in param_grid['alpha']:
            for lr in param_grid['learning_rate']:
                nn = MLPClassifier(hidden_layer_sizes=hidden_layer,
                                   activation='relu',
                                   solver='adam',
                                   alpha=alpha,
                                   learning_rate=lr,
                                   max_iter=1,
                                   warm_start=True,
                                   random_state=42)
                nn.fit(X_train, Y_train)
                Y_val_pred = nn.predict(X_valid)
                metrics[(hidden_layer, alpha, lr)] = {
                    "Accuracy": accuracy_score(Y_valid, Y_val_pred),
                    "F1": f1_score(Y_val_pred, Y_valid, average='macro')
                }

    # Pick best model according to validation F1
    best_params = max(metrics, key=lambda x: metrics[x]['F1'])
    best_nn = MLPClassifier(hidden_layer_sizes=best_params[0],
                            activation='relu',
                            solver='adam',
                            alpha=best_params[1],
                            learning_rate=best_params[2],
                            max_iter=1,
                            warm_start=True,
                            random_state=42)

    # Record metrics over iterations
    iterations = 500
    loss_list, acc_list, f1_list = [], [], []

    for i in range(iterations):
        best_nn.fit(X_train, Y_train)
        Y_pred = best_nn.predict(X_train)
        loss_list.append(best_nn.loss_)
        acc_list.append(accuracy_score(Y_train, Y_pred))
        f1_list.append(f1_score(Y_train, Y_pred, average='macro'))

    # Save metrics for visualization file
    with open("nn_training_metrics.pkl", "wb") as f:
        pickle.dump((loss_list, acc_list, f1_list), f)

    # Retrains the svm model using both the training and validation data combined,
    # so it can learn from all available labeled examples before final evaluation or saving.
    best_nn.fit(
        np.vstack([X_train, X_valid]),
        np.hstack([Y_train, Y_valid])
    )

    # Save the best model
    with open("neural_net_model.pkl", "wb") as f:
        pickle.dump(best_nn, f)

    print(f"Neural Network - best params (from validation): (hidden_layer={best_params[0]}, alpha={best_params[1]}, lr={best_params[2]})")
    return best_nn

# Train all models and save pickles
models = {
    'knn': train_knn(),
    'naive_bayes': train_nb(),
    'decision_tree': train_dt(),
    'svm': train_svm(),
    'neural_net': train_nn()
}

for name, model in models.items():
    with open(f"{name}_model.pkl", "wb") as f:
        pickle.dump(model, f)

# Pick best model based on F1-score
best_score = 0
best_model = None
for name, model in models.items():
    Y_pred = model.predict(X_test)
    score = f1_score(Y_test, Y_pred, average='macro')
    if score > best_score:
        best_score = score
        best_model = model
    print(f"{name}: {score}")
print(f"\nbest model: {best_model} with score of {best_score}")

# Save best model
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)