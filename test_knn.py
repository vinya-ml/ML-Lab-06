import numpy as np
from collections import Counter

from knn_impl import (
    load_dataset,
    encode_categorical,
    impute_missing_values,
    bubble_sort,
    insertion_sort,
    selection_sort,
    calculate_distance,
    find_k_nearest_neighbors_distances,
    majority_vote,
    weighted_majority_vote,
    KNNClassifier,
    train_test_split_data,
)


# ============================================================
# Dataset Loading Tests
# ============================================================

def test_load_dataset():
    """Test dataset loading from CSV."""
    data = load_dataset(r"C:\Vinya\Sem - 5\ML\Certifications\mall_customers.csv")
    assert data is not None
    assert len(data) > 0
    assert list(data.columns) is not None


# ============================================================
# Categorical Encoding Tests
# ============================================================

def test_encode_categorical():
    """Test categorical column encoding."""
    data = load_dataset(r"C:\Vinya\Sem - 5\ML\Certifications\mall_customers.csv")
    encoded_data, mapping = encode_categorical(data, 'Gender')
    assert 'Gender' in encoded_data.columns
    assert len(mapping) == len(encoded_data['Gender'].unique())
    # Verify encoding was applied
    assert set(encoded_data['Gender'].unique()).issubset({0, 1})


# ============================================================
# Missing Value Imputation Tests
# ============================================================

def test_impute_missing_values_mean():
    """Test mean imputation on numerical columns."""
    data = load_dataset(r"C:\Vinya\Sem - 5\ML\Certifications\mall_customers.csv")
    # Encode categorical first, then impute
    data, _ = encode_categorical(data, 'Gender')
    data_copy = data.copy()
    # Only impute numerical columns
    num_cols = [c for c in data.columns if data[c].dtype in ['float64', 'int64']]
    if num_cols:
        imputed = impute_missing_values(data_copy, strategy='mean', columns=num_cols)
        # No NaN values should remain in imputed columns
        assert imputed[num_cols].isnull().sum().sum() == 0


def test_impute_missing_values_median():
    """Test median imputation on numerical columns."""
    data = load_dataset(r"C:\Vinya\Sem - 5\ML\Certifications\mall_customers.csv")
    # Encode categorical first, then impute
    data, _ = encode_categorical(data, 'Gender')
    data_copy = data.copy()
    # Only impute numerical columns
    num_cols = [c for c in data.columns if data[c].dtype in ['float64', 'int64']]
    if num_cols:
        imputed = impute_missing_values(data_copy, strategy='median', columns=num_cols)
        # No NaN values should remain in imputed columns
        assert imputed[num_cols].isnull().sum().sum() == 0


def test_impute_missing_values_mode():
    """Test mode imputation."""
    data = load_dataset(r"C:\Vinya\Sem - 5\ML\Certifications\mall_customers.csv")
    # Encode categorical first, then impute
    data, _ = encode_categorical(data, 'Gender')
    data_copy = data.copy()
    # Only impute numerical columns
    num_cols = [c for c in data.columns if data[c].dtype in ['float64', 'int64']]
    if num_cols:
        imputed = impute_missing_values(data_copy, strategy='mode', columns=num_cols)
        # No NaN values should remain in imputed columns
        assert imputed[num_cols].isnull().sum().sum() == 0


# ============================================================
# Sorting Algorithm Tests
# ============================================================

def test_bubble_sort():
    """Test bubble sort."""
    arr = [5, 3, 8, 1, 2]
    sorted_arr = bubble_sort(arr)
    assert sorted_arr == [1, 2, 3, 5, 8]


def test_insertion_sort():
    """Test insertion sort."""
    arr = [5, 3, 8, 1, 2]
    sorted_arr = insertion_sort(arr)
    assert sorted_arr == [1, 2, 3, 5, 8]


def test_selection_sort():
    """Test selection sort."""
    arr = [5, 3, 8, 1, 2]
    sorted_arr = selection_sort(arr)
    assert sorted_arr == [1, 2, 3, 5, 8]


# ============================================================
# Distance Calculation Tests
# ============================================================

def test_calculate_euclidean_distance():
    """Test Euclidean distance calculation."""
    p1 = [1, 2, 3]
    p2 = [4, 6, 8]
    dist = calculate_distance(p1, p2, 'euclidean')
    expected = np.sqrt((4-1)**2 + (6-2)**2 + (8-3)**2)
    assert abs(dist - expected) < 1e-10


def test_calculate_manhattan_distance():
    """Test Manhattan distance calculation."""
    p1 = [1, 2, 3]
    p2 = [4, 6, 8]
    dist = calculate_distance(p1, p2, 'manhattan')
    expected = np.abs(4-1) + np.abs(6-2) + np.abs(8-3)
    assert abs(dist - expected) < 1e-10


def test_calculate_minkowski_distance():
    """Test Minkowski distance calculation (p=3)."""
    p1 = [1, 2, 3]
    p2 = [4, 6, 8]
    dist = calculate_distance(p1, p2, 'minkowski')
    expected = np.sum(np.abs(np.array(p1) - np.array(p2)) ** 3) ** (1/3)
    assert abs(dist - expected) < 1e-10


# ============================================================
# k-NN Neighbor Finding Tests
# ============================================================

def test_find_k_nearest_neighbors():
    """Test finding k nearest neighbors."""
    train_features = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    test_feature = np.array([1, 2])
    k_neighbors = find_k_nearest_neighbors_distances(train_features, test_feature, k=2)
    assert len(k_neighbors) == 2
    # Should include the test point itself at distance 0
    assert k_neighbors[0][0] == 0.0


# ============================================================
# Majority Vote Tests
# ============================================================

def test_majority_vote():
    """Test majority voting with tie-breaking."""
    # Clear majority
    neighbors = [(0, 0), (0, 1), (0, 2)]  # (distance, label)
    label = majority_vote(neighbors)
    assert label == 0
    
    # Tie-breaking: choose smallest label
    neighbors = [(0, 1), (0, 0), (0, 2)]  # tie between 0, 1, 2
    label = majority_vote(neighbors)
    assert label == 0  # smallest label wins


# ============================================================
# Weighted Majority Vote Tests
# ============================================================

def test_weighted_majority_vote():
    """Test weighted majority voting."""
    neighbors = [(0.5, 0), (0.5, 1), (1.0, 0)]
    label = weighted_majority_vote(neighbors)
    # Label 0 has weight 1/0.5 + 1/1 = 2 + 1 = 3
    # Label 1 has weight 1/0.5 = 2
    # So 0 should win
    assert label == 0


# ============================================================
# KNNClassifier Tests
# ============================================================

def test_knn_classifier_fit():
    """Test KNN classifier fit method."""
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 0, 1, 1])
    knn = KNNClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    assert knn.X_train is not None
    assert knn.y_train is not None


def test_knn_classifier_predict():
    """Test KNN classifier predict method."""
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 0, 1, 1])
    knn = KNNClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    X_test = np.array([[1, 2], [5, 6]])
    predictions = knn.predict(X_test)
    assert len(predictions) == 2


def test_knn_classifier_score():
    """Test KNN classifier score method."""
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 0, 1, 1])
    knn = KNNClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    X_test = np.array([[1, 2], [5, 6]])
    y_test = np.array([0, 1])
    accuracy = knn.score(X_test, y_test)
    assert 0 <= accuracy <= 1


def test_knn_classifier_default_params():
    """Test KNN classifier with default parameters."""
    knn = KNNClassifier()
    assert knn.n_neighbors == 3
    assert knn.distance_metric == 'euclidean'
    assert knn.sorting_algorithm == 'bubble'
    assert knn.weighting is False


def test_knn_classifier_weighting():
    """Test KNN classifier with weighting enabled."""
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 0, 1, 1])
    knn = KNNClassifier(n_neighbors=3, weighting=True)
    knn.fit(X_train, y_train)
    X_test = np.array([[1, 2]])
    prediction = knn.predict(X_test)
    assert prediction is not None


# ============================================================
# Train/Test Split Tests
# ============================================================

def test_train_test_split():
    """Test train/test split function."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    y = np.array([0, 0, 1, 1, 0, 1])
    X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=0.5)
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)


# ============================================================
# Performance Measurement Tests
# ============================================================

def test_measure_performance():
    """Test performance measurement."""
    from knn_impl import measure_performance
    import time
    
    X_train = np.random.rand(20, 3)
    y_train = np.random.randint(0, 2, 20)
    X_test = np.random.rand(5, 3)
    y_test = np.random.randint(0, 2, 5)
    
    knn = KNNClassifier(n_neighbors=3)
    result = measure_performance(knn, X_test, y_test, runs=3)
    
    assert 'accuracy_mean' in result
    assert 'accuracy_std' in result
    assert 'time_mean' in result
    assert 'time_std' in result
    assert 0 <= result['accuracy_mean'] <= 1
    assert result['time_mean'] >= 0


# Run all tests if executed directly
if __name__ == "__main__":
    import sys
    
    # Get list of test functions to run
    test_functions = [func for name, func in sorted(globals().items())
                      if name.startswith('test_') and callable(func)]
    
    failed = []
    passed = []
    
    for test_func in test_functions:
        try:
            test_func()
            passed.append(test_func.__name__)
        except Exception as e:
            failed.append((test_func.__name__, str(e)))
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Test Results: {len(passed)} passed, {len(failed)} failed")
    print(f"{'='*50}")
    
    for test_name in passed:
        print(f"  PASS: {test_name}")
    
    for test_name, error in failed:
        print(f"  FAIL: {test_name} - {error}")
    
    # Exit with error code if any failures
    if failed:
        print(f"\n{len(failed)} test(s) failed!")
        sys.exit(1)
    else:
        print(f"\nAll {len(passed)} tests passed!")