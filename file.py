#Imports:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import pickle
from sklearn.metrics import f1_score

#Load and Preview Dataset:
cols = ["fLength", "fWidth", "fSize", "fConc", "fConcl", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv("../ML - MAGICGammaTelescope/magic04.data", names=cols)
#print(df.head())

#Encode Target Variable:
df["class"] = (df["class"] == "g").astype(int)
#print(df.head())

#Feature Distributions by Class:
"""
for label in cols[:-1]:
    plt.hist(df[df["class"]==1][label], color="blue", label="gamma", alpha=0.7, density=True)
    plt.hist(df[df["class"]==0][label], color="red", label="hadron", alpha=0.7, density=True)
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()
"""

#Split Dataset into Train, Validation, and Test Sets:
#train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])
train, valid, test = np.split(df.sample(frac=1, random_state=42),[int(0.6 * len(df)), int(0.8 * len(df))])

#Scale and Optionally Oversample Dataset:
def scale_dataset(dataframe, scaler=None, oversample=False, fit=True):
    X = dataframe[dataframe.columns[:-1]].values
    Y = dataframe[dataframe.columns[-1]].values
    if fit:
        scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    if oversample:
        ros = RandomOverSampler()
        X, Y = ros.fit_resample(X, Y)
    data = np.hstack((X, np.reshape(Y, (-1, 1))))
    return data, X, Y, scaler

#Apply Scaling and Oversampling to Splits:
train, X_train, Y_train, scaler = scale_dataset(train, oversample=True, fit=True)
valid, X_valid, Y_valid, _ = scale_dataset(valid, scaler=scaler, fit=False)
test, X_test, Y_test, _ = scale_dataset(test, scaler=scaler, fit=False)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

"""def KNN():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report
    from sklearn.model_selection import GridSearchCV

    param_grid = {
        'n_neighbors': [1, 3, 5, 7, 9, 11, 15, 21, 31, 51, 71, 91],
        'p': [1, 2]
    }
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5)
    grid_search.fit(X_train, Y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_
    Y_pred = best_model.predict(X_test)
    print(classification_report(Y_test, Y_pred))

    return best_model

def Naive_Bayes():
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import classification_report
    from sklearn.model_selection import GridSearchCV

    param_grid = {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]  # Smoothing parameter
    }
    nb = GaussianNB()
    grid_search = GridSearchCV(nb, param_grid, cv=5)
    grid_search.fit(X_train, Y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_
    Y_pred = best_model.predict(X_test)
    print(classification_report(Y_test, Y_pred))

    return best_model

def Decision_tree():
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.metrics import classification_report
    from sklearn.model_selection import GridSearchCV

    param_grid = {
        'criterion': ['gini', 'entropy'],#Function to measure the quality of a split
        'max_depth': [None, 3, 4, 5, 6],
        'min_samples_split': [4, 5, 6],#Minimum number of samples required to split an internal node
        'min_samples_leaf': [2, 3, 4]#Minimum number of samples required to be at a leaf node
    }
    dt = DecisionTreeClassifier()
    grid_search = GridSearchCV(dt, param_grid, cv=5)
    grid_search.fit(X_train, Y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_
    Y_pred = best_model.predict(X_test)
    print(classification_report(Y_test, Y_pred))

    return best_model

def SVM():
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report
    from sklearn.model_selection import GridSearchCV

    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': [0.1, 0.01],
        'kernel': ['linear', 'rbf']
    }
    svm_model = SVC()
    grid_search = GridSearchCV(svm_model, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, Y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_
    Y_pred = best_model.predict(X_test)
    print(classification_report(Y_test, Y_pred))

    return best_model

def Neural_Network():
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import classification_report
    from sklearn.model_selection import GridSearchCV

    param_grid = {
        'hidden_layer_sizes': [(50,), (100,)],
        'activation': ['relu'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.001],
        'learning_rate': ['constant', 'adaptive']
    }
    mlp = MLPClassifier(max_iter=1000)
    grid_search = GridSearchCV(mlp, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, Y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_
    Y_pred = best_model.predict(X_test)
    print(classification_report(Y_test, Y_pred))

    return best_model

best_knn = KNN()
best_nb = Naive_Bayes()
best_dt = Decision_tree()
best_svm = SVM()
best_nn = Neural_Network()

models = {
    'knn': best_knn,
    'naive_bayes': best_nb,
    'decision_tree': best_dt,
    'svm': best_svm,
    'neural_net': best_nn
}

for name, model in models.items():
    with open(f"{name}_model.pkl", "wb") as f:
        pickle.dump(model, f)

# --- 3. Load the scaler and models later ---
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

model_files = {
    'knn': 'knn_model.pkl',
    'naive_bayes': 'naive_bayes_model.pkl',
    'decision_tree': 'decision_tree_model.pkl',
    'svm': 'svm_model.pkl',
    'neural_net': 'neural_net_model.pkl'
}

loaded_models = {}
for name, file in model_files.items():
    with open(file, "rb") as f:
        loaded_models[name] = pickle.load(f)

# --- 4. Evaluate all models and pick the best one ---
best_score = 0
best_model_name = ""
best_model = None

for name, model in loaded_models.items():
    Y_pred = model.predict(X_test)  # make sure X_test is scaled!
    score = f1_score(Y_test, Y_pred)
    print(f"{name}: F1-score = {score:.4f}")

    if score > best_score:
        best_score = score
        best_model_name = name
        best_model = model

print("\nBest model:", best_model_name, "with F1-score =", round(best_score, 4))

# --- 5. Save the best model separately ---
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)"""


# Load scaler and best model
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("best_model.pkl", "rb") as f:
    best_model = pickle.load(f)

# Example new sample
new_sample = [[20.5, 1.5, 120, 0.5, 0.3, 0.02, 0.1, 0.05, 1.2, 0.8]]
new_sample_scaled = scaler.transform(new_sample)

# Predict
prediction = best_model.predict(new_sample_scaled)
print("Predicted class:", prediction[0])