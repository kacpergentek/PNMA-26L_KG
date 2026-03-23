import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE

def train_test_split_custom(X, y, ratio=0.7):
    np.random.seed(42)
    indices = np.random.permutation(X.shape[0])
    train_len = int(X.shape[0] * ratio)
    
    X_train = X[indices[:train_len]]
    y_train = y[indices[:train_len]]
    X_test = X[indices[train_len:]]
    y_test = y[indices[train_len:]]
    
    return X_train, X_test, y_train, y_test

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def custom_knn(X_train, y_train, X_test, k):
    def classify_single(x):
        dists = [euclidean_distance(x, i) for i in X_train]
        indices = np.argpartition(dists, k)[:k]
        return np.argmax(np.bincount(y_train[indices]))

    return np.array([classify_single(x) for x in X_test])

def calculate_metrics(y_true, y_pred):
    accuracy = np.mean(y_true == y_pred)
    classes = np.unique(y_true)
    
    print(f"Ogólne Accuracy: {accuracy:.4f}\n")
    
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"Klasa {c}:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-score:  {f1:.4f}\n")

iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

print(f"Liczba próbek: {X.shape[0]}")
print(f"Liczba cech: {X.shape[1]}")
print(f"Dostępne klasy: {target_names}\n")

X_train, X_test, y_train, y_test = train_test_split_custom(X, y, ratio=0.7)

k_neighbors = 3
y_pred = custom_knn(X_train, y_train, X_test, k_neighbors)

calculate_metrics(y_test, y_pred)

tsne = TSNE(n_components=2, random_state=42)
X_test_2d = tsne.fit_transform(X_test)

plt.figure(figsize=(8, 6))
scatter = sns.scatterplot(x=X_test_2d[:, 0], y=X_test_2d[:, 1], hue=y_pred, palette='viridis', s=100)

handles, _ = scatter.get_legend_handles_labels()
plt.legend(handles=handles, labels=list(target_names), title="Przewidziana klasa")

plt.title('Wizualizacja t-SNE po klasyfikacji algorytmem KNN')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.grid(True)
plt.show()