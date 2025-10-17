import pandas as pd
import pickle
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import warnings
from sklearn.exceptions import ConvergenceWarning

def plot_feature_distributions():
    cols = ["fLength", "fWidth", "fSize", "fConc", "fConcl", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
    df = pd.read_csv("../ML - MAGICGammaTelescope/magic04.data", names=cols)
    df["class"] = (df["class"] == "g").astype(int)

    cols = df.columns
    for label in cols[:-1]:  # skip the last column (class)
        plt.figure(figsize=(6,4))
        plt.hist(df[df["class"]==1][label], color="blue", label="Gamma", alpha=0.7, density=True)
        plt.hist(df[df["class"]==0][label], color="red", label="Hadron", alpha=0.7, density=True)
        plt.title(f"Distribution of {label}")
        plt.xlabel(label)
        plt.ylabel("Probability Density")
        plt.legend()
        plt.tight_layout()
        plt.show()

def plot_class_balance():
    cols = ["fLength", "fWidth", "fSize", "fConc", "fConcl", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
    df = pd.read_csv("../ML - MAGICGammaTelescope/magic04.data", names=cols)
    df["class"] = (df["class"] == "g").astype(int)  # 1=Gamma, 0=Hadron

    # Original class counts
    before_counts = df["class"].value_counts()

    # Oversample to balance classes
    X = df[df.columns[:-1]].values
    y = df["class"].values
    ros = RandomOverSampler()
    _, y_res = ros.fit_resample(X, y)
    after_counts = pd.Series(y_res).value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(before_counts.index.astype(str), before_counts.values, color=["red", "blue"])
    axes[0].set_title("Before Oversampling")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Count")
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["Hadron (0)", "Gamma (1)"])

    axes[1].bar(after_counts.index.astype(str), after_counts.values, color=["red", "blue"])
    axes[1].set_title("After Oversampling")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Count")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Hadron (0)", "Gamma (1)"])

    plt.suptitle("Class Balance Before and After Oversampling", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_model_performance_from_test():
    model_names = ["knn", "naive_bayes", "decision_tree", "svm", "neural_net"]
    with open("test_data.pkl", "rb") as f:
        X_test, Y_test = pickle.load(f)

    models = {}
    for name in model_names:
        with open(f"{name}_model.pkl", "rb") as f:
            models[name] = pickle.load(f)

    scores = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        scores[name] = {
            "F1": f1_score(Y_test, y_pred, average="macro"),
            "Accuracy": accuracy_score(Y_test, y_pred)
        }

    df_scores = pd.DataFrame(scores).T
    df_scores.plot(kind="bar", figsize=(8, 5), color=['red', 'blue'])
    plt.title("Model Performance Comparison (on test set)")
    plt.ylabel("Score")
    plt.ylim(0.6, 0.9)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix_best():
    with open("best_model.pkl", "rb") as f:
        best_model = pickle.load(f)
    with open("test_data.pkl", "rb") as f:
        X_test, Y_test = pickle.load(f)

    Y_pred = best_model.predict(X_test)
    cm = confusion_matrix(Y_test, Y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Hadron (0)', 'Gamma (1)'],
                yticklabels=['Hadron (0)', 'Gamma (1)'])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix of Best Model")
    plt.tight_layout()
    plt.show()

def plot_knn_metrics():
    # Load metrics
    with open("knn_metrics.pkl", "rb") as f:
        metrics = pickle.load(f)

    # Load the best KNN model (selected based on validation performance) to retrieve its chosen K and p values
    with open("knn_model.pkl", "rb") as f:
        best_knn = pickle.load(f)
    best_k = best_knn.n_neighbors
    best_p = best_knn.p

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for idx, p_val in enumerate(sorted(set(p for _, p in metrics.keys()))):
        ax = axes[idx]
        ks = sorted([k for k, p in metrics.keys() if p == p_val])
        accs = [metrics[(k, p_val)]["Accuracy"] for k in ks]
        f1s = [metrics[(k, p_val)]["F1"] for k in ks]

        ax.plot(ks, accs, marker='o', label="Accuracy")
        ax.plot(ks, f1s, marker='s', label="F1 Score")
        ax.set_title(f"KNN Metrics (p={p_val})")
        ax.set_xlabel("K")
        if idx == 0:
            ax.set_ylabel("Score")
        else:
            ax.set_ylabel("Score")
        ax.set_xticks(ks)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Highlight the K chosen based on validation performance
        if p_val == best_p:
            best_idx = ks.index(best_k)
            ax.axvline(x=best_k, color='red', linestyle='--', label=f"Chosen K={best_k}")
            ax.scatter(best_k, accs[best_idx], color='red', s=100, edgecolor='k')
            ax.scatter(best_k, f1s[best_idx], color='red', s=100, edgecolor='k')

    plt.tight_layout()
    plt.show()

def plot_nn_training_metrics():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    with open("nn_training_metrics.pkl", "rb") as f:
        loss_list, acc_list, f1_list = pickle.load(f)

    plt.figure(figsize=(10,6))
    plt.plot(loss_list, label='Loss', color='green', linewidth=4)
    plt.plot(acc_list, label='Accuracy', color='red', linewidth=10)
    plt.plot(f1_list, label='F1 Score', color='blue', linewidth=4)
    plt.title("Neural Network Training Metrics")
    plt.xlabel("Iteration")
    plt.ylabel("Metric Value")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

#Visualizations

plot_feature_distributions()

plot_class_balance()

plot_model_performance_from_test()

plot_confusion_matrix_best()

plot_knn_metrics()

plot_nn_training_metrics()