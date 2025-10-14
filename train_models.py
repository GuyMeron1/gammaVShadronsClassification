# Imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

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
    param_grid = {'n_neighbors': [1,3,5,7,9,11,15,21,31,51,71,91], 'p':[1,2]}
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid.fit(X_train, Y_train)
    return grid.best_estimator_
def train_nb():
    param_grid = {'var_smoothing': [1e-9,1e-8,1e-7,1e-6]}
    grid = GridSearchCV(GaussianNB(), param_grid, cv=5)
    grid.fit(X_train, Y_train)
    return grid.best_estimator_
def train_dt():
    param_grid = {'criterion':['gini','entropy'],
                  'max_depth':[None,3,4,5,6],
                  'min_samples_split':[4,5,6],
                  'min_samples_leaf':[2,3,4]}
    grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
    grid.fit(X_train, Y_train)
    return grid.best_estimator_
def train_svm():
    param_grid = {'C':[0.1,1,10],
                  'gamma':[0.1,0.01],
                  'kernel':['linear','rbf']}
    grid = GridSearchCV(SVC(probability=True), param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train, Y_train)
    return grid.best_estimator_
def train_nn():
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,)],
        'activation': ['relu'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.001],
        'learning_rate': ['constant', 'adaptive']
    }
    grid = GridSearchCV(
        MLPClassifier(max_iter=1, warm_start=True, random_state=42),
        param_grid,
        cv=5,
        n_jobs=-1
    )
    grid.fit(X_train, Y_train)
    best_nn = grid.best_estimator_

    # Record training metrics for visualization file
    iterations = 500
    loss_list, acc_list, f1_list = [], [], []

    for i in range(iterations):
        best_nn.fit(X_train, Y_train)
        Y_pred = best_nn.predict(X_train)
        loss_list.append(best_nn.loss_)
        acc_list.append(accuracy_score(Y_train, Y_pred))
        f1_list.append(f1_score(Y_train, Y_pred, average='macro'))

    with open("nn_training_metrics.pkl", "wb") as f:
        pickle.dump((loss_list, acc_list, f1_list), f)

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
    if isinstance(model, tuple):
        model, _ = model
    Y_pred = model.predict(X_test)
    score = f1_score(Y_test, Y_pred, average='macro')
    if score > best_score:
        best_score = score
        best_model = model
    print(f"{name}: {score}")
    print(classification_report(Y_test, Y_pred))
print(f"best model: {best_model} with score of {best_score}")

# Save best model
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)