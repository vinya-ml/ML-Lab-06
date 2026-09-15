import numpy as np
import pandas as pd
import time
import tracemalloc
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier as SKLKNN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ============================================================
# Version 1: Custom k-NN (code written by me)
# ============================================================

def load_data():
    """Load and preprocess dataset."""
    file_path = r"C:\Vinya\Sem - 5\ML\Certifications\mall_customers.csv"
    data = pd.read_csv(file_path)
    data = data.copy()
    
    # Encode categorical column (Gender)
    unique_values = data['Gender'].unique()
    mapping = {val: idx for idx, val in enumerate(unique_values)}
    data['Gender'] = data['Gender'].map(mapping)
    
    # Impute missing values
    data = data.fillna(data.mean(numeric_only=True))
    
    # Prepare features and target
    X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
    y = data['Gender'].values
    
    # Split into train and test
    np.random.seed(42)
    n_samples = X.shape[0]
    n_test = int(n_samples * 0.3)
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test


def calculate_distance(point1, point2, metric='euclidean'):
    """Calculate distance between two points."""
    p1 = np.array(point1, dtype=float)
    p2 = np.array(point2, dtype=float)
    if metric == 'euclidean':
        return np.sqrt(np.sum((p1 - p2) ** 2))
    elif metric == 'manhattan':
        return np.sum(np.abs(p1 - p2))
    elif metric == 'minkowski':
        p = 3
        return np.sum(np.abs(p1 - p2) ** p) ** (1 / p)
    else:
        raise ValueError("Metric must be 'euclidean', 'manhattan', or 'minkowski'")


def find_k_nearest_neighbors(train_features, test_feature, k, distance_metric='euclidean', sorting_algo='bubble'):
    """Find k nearest neighbors."""
    distances = []
    for i, train_feat in enumerate(train_features):
        dist = calculate_distance(test_feature, train_feat, distance_metric)
        distances.append((dist, i))
    
    if sorting_algo == 'bubble':
        dist_copy = distances.copy()
        n = len(dist_copy)
        for i in range(n):
            for j in range(0, n - i - 1):
                if dist_copy[j][0] > dist_copy[j + 1][0]:
                    dist_copy[j], dist_copy[j + 1] = dist_copy[j + 1], dist_copy[j]
        sorted_dist = dist_copy
    elif sorting_algo == 'insertion':
        dist_copy = distances.copy()
        for i in range(len(dist_copy)):
            key = dist_copy[i]
            j = i - 1
            while j >= 0 and dist_copy[j][0] > key[0]:
                dist_copy[j + 1] = dist_copy[j]
                j -= 1
            dist_copy[j + 1] = key
        sorted_dist = dist_copy
    elif sorting_algo == 'selection':
        dist_copy = distances.copy()
        n = len(dist_copy)
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if dist_copy[j][0] < dist_copy[min_idx][0]:
                    min_idx = j
            sorted_dist = dist_copy.pop(min_idx) if i < n - 1 else dist_copy
        if dist_copy:
            sorted_dist.extend(dist_copy)
    else:
        sorted_dist = sorted(distances, key=lambda x: x[0])
    
    return sorted_dist[:k]


def majority_vote(neighbors):
    """Determine class label via majority voting."""
    labels = [label for (_, label) in neighbors]
    count = Counter(labels)
    max_count = max(count.values())
    tied_labels = [label for label, cnt in count.items() if cnt == max_count]
    return min(tied_labels)


def predict_custom(X_train, y_train, X_test, k=3, metric='euclidean', sorting_algo='bubble'):
    """Predict using custom k-NN."""
    predictions = []
    for test_point in X_test:
        k_neighbors = find_k_nearest_neighbors(X_train, test_point, k, metric, sorting_algo)
        labels = [y_train[idx] for (_, idx) in k_neighbors]
        prediction = majority_vote(list(zip(k_neighbors, labels)))
        predictions.append(prediction)
    return np.array(predictions)


# ============================================================
# Version 2: Scikit-Learn k-NN (inbuilt functions)
# ============================================================

def predict_sklearn(X_train, y_train, X_test, n_neighbors=3):
    """Predict using sklearn k-NN."""
    knn = SKLKNN(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn.predict(X_test)


# ============================================================
# Version 3: GenAI-generated k-NN code
# ============================================================

def calculate_distance_genai(point1, point2, metric='euclidean'):
    """Distance calculation from GenAI-generated code."""
    p1 = np.array(point1, dtype=float)
    p2 = np.array(point2, dtype=float)
    if metric == 'euclidean':
        return np.sqrt(np.sum((p1 - p2) ** 2))
    elif metric == 'manhattan':
        return np.sum(np.abs(p1 - p2))
    elif metric == 'minkowski':
        p = 3
        return np.sum(np.abs(p1 - p2) ** p) ** (1 / p)
    else:
        raise ValueError("Metric must be 'euclidean', 'manhattan', or 'minkowski'")


def find_k_nearest_neighbors_genai(train_features, test_feature, k, distance_metric='euclidean', sorting_algo='bubble'):
    """Find k nearest neighbors from GenAI-generated code."""
    distances = []
    for i, train_feat in enumerate(train_features):
        dist = calculate_distance_genai(test_feature, train_feat, distance_metric)
        distances.append((dist, i))
    
    if sorting_algo == 'bubble':
        sorted_dist = []
        dist_copy = distances.copy()
        n = len(dist_copy)
        for i in range(n):
            for j in range(0, n - i - 1):
                if dist_copy[j][0] > dist_copy[j + 1][0]:
                    dist_copy[j], dist_copy[j + 1] = dist_copy[j + 1], dist_copy[j]
        sorted_dist = dist_copy
    elif sorting_algo == 'insertion':
        sorted_dist = []
        dist_copy = distances.copy()
        for i in range(len(dist_copy)):
            key = dist_copy[i]
            j = i - 1
            while j >= 0 and dist_copy[j][0] > key[0]:
                dist_copy[j + 1] = dist_copy[j]
                j -= 1
            dist_copy[j + 1] = key
        sorted_dist = dist_copy
    elif sorting_algo == 'selection':
        sorted_dist = []
        dist_copy = distances.copy()
        n = len(dist_copy)
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if dist_copy[j][0] < dist_copy[min_idx][0]:
                    min_idx = j
            sorted_dist.append(dist_copy.pop(min_idx))
        if dist_copy:
            sorted_dist.extend(dist_copy)
    else:
        sorted_dist = sorted(distances, key=lambda x: x[0])
    
    return sorted_dist[:k]


def majority_vote_genai(neighbors):
    """Majority vote from GenAI-generated code."""
    labels = [label for (_, label) in neighbors]
    count = Counter(labels)
    max_count = max(count.values())
    tied_labels = [label for label, cnt in count.items() if cnt == max_count]
    return min(tied_labels)


def predict_genai(X_train, y_train, X_test, k=3, metric='euclidean', sorting_algo='bubble'):
    """Predict using GenAI-generated k-NN code."""
    predictions = []
    for test_point in X_test:
        k_neighbors = find_k_nearest_neighbors_genai(X_train, test_point, k, metric, sorting_algo)
        k_labels = [y_train[idx] for (_, idx) in k_neighbors]
        prediction = majority_vote_genai(list(zip(k_neighbors, k_labels)))
        predictions.append(prediction)
    return np.array(predictions)


# ============================================================
# Evaluation Metrics
# ============================================================

def compute_metrics(y_true, y_pred):
    """Compute accuracy, precision, recall, f-score."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# ============================================================
# Performance Measurement
# ============================================================

def measure_permission(X_train, X_test, y_train, y_test, k=3, runs=10):
    """Measure performance across multiple runs."""
    results = {}
    
    # Version 1: Custom k-NN
    custom_times = []
    custom_y_pred = None
    for _ in range(runs):
        tracemalloc.start()
        start = time.time()
        y_pred = predict_custom(X_train, y_train, X_test, k=k)
        elapsed = time.time() - start
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        custom_times.append(elapsed)
        if custom_y_pred is None:
            custom_y_pred = y_pred
    
    results['custom'] = {
        'accuracy': np.mean([compute_metrics(y_test, custom_y_pred)['accuracy']]),
        'precision': np.mean([compute_metrics(y_test, custom_y_pred)['precision']]),
        'recall': np.mean([compute_metrics(y_test, custom_y_pred)['recall']]),
        'f1': np.mean([compute_metrics(y_test, custom_y_pred)['f1']]),
        'time_mean': np.mean(custom_times),
        'time_std': np.std(custom_times)
    }
    
    # Version 2: Scikit-Learn k-NN
    sklearn_times = []
    sklearn_y_pred = None
    for _ in range(runs):
        start = time.time()
        y_pred = predict_sklearn(X_train, y_train, X_test, n_neighbors=k)
        elapsed = time.time() - start
        sklearn_times.append(elapsed)
        if sklearn_y_pred is None:
            sklearn_y_pred = y_pred
    
    results['sklearn'] = {
        'accuracy': compute_metrics(y_test, sklearn_y_pred)['accuracy'],
        'precision': compute_metrics(y_test, sklearn_y_pred)['precision'],
        'recall': compute_metrics(y_test, sklearn_y_pred)['recall'],
        'f1': compute_metrics(y_test, sklearn_y_pred)['f1'],
        'time_mean': np.mean(sklearn_times),
        'time_std': np.std(sklearn_times)
    }
    
    # Version 3: GenAI k-NN
    genai_times = []
    genai_y_pred = None
    for _ in range(runs):
        tracemalloc.start()
        start = time.time()
        y_pred = predict_genai(X_train, y_train, X_test, k=k)
        elapsed = time.time() - start
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        genai_times.append(elapsed)
        if genai_y_pred is None:
            genai_y_pred = y_pred
    
    results['genai'] = {
        'accuracy': np.mean([compute_metrics(y_test, genai_y_pred)['accuracy']]),
        'precision': np.mean([compute_metrics(y_test, genai_y_pred)['precision']]),
        'recall': np.mean([compute_metrics(y_test, genai_y_pred)['recall']]),
        'f1': np.mean([compute_metrics(y_test, genai_y_pred)['f1']]),
        'time_mean': np.mean(genai_times),
        'time_std': np.std(genai_times)
    }
    
    return results


# ============================================================
# Main
# ============================================================

def main():
    """Run performance comparison."""
    X_train, X_test, y_train, y_test = load_data()
    
    print("=" * 70)
    print("K-NN ALGORITHM PERFORMANCE COMPARISON")
    print("=" * 70)
    print(f"\nDataset: mall_customers.csv")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Unique classes: {len(np.unique(y_test))}")
    print(f"k value: 3")
    print(f"Distance metric: euclidean")
    print(f"Sorting algorithm: bubble")
    print("-" * 70)
    
    results = measure_permission(X_train, X_test, y_train, y_test, k=3, runs=10)
    
    # Print results table
    print("\n{:<15} {:<12} {:<12} {:<12} {:<12} {:<15} {:<15}".format(
        'Version', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Time (s)', 'Time STD'))
    print("-" * 70)
    
    for version, metrics in results.items():
        print("{:<15} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<15.6f} {:<15.6f}".format(
            version.capitalize(),
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1'],
            metrics['time_mean'],
            metrics['time_std']
        ))
    
    print("-" * 70)
    
    # Summary
    print("\nSUMMARY:")
    print(f"  Most accurate: {max(results, key=lambda v: results[v]['accuracy'])}")
    print(f"  Fastest: {min(results, key=lambda v: results[v]['time_mean'])}")
    print(f"  Most stable (lowest time STD): {min(results, key=lambda v: results[v]['time_std'])}")


if __name__ == "__main__":
    main()