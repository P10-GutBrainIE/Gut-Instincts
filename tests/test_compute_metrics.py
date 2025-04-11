import numpy as np
import pytest
from NER.compute_metrics import compute_metrics


@pytest.fixture
def perfect_prediction_data():
    predictions = np.array([[[0.1, 0.9, 0.89, 0.05, 0.78], [0.8, 0.2, 0.5, 0.7, 0.7]]])
    labels = np.array([[1, 0]])
    return predictions, labels


@pytest.fixture
def partial_prediction_data():
    predictions = np.array([[[0.6, 0.4, 0.89, 0.05, 0.78], [0.3, 0.7, 0.89, 0.05, 0.78]]])
    labels = np.array([[2, 4]])
    return predictions, labels


@pytest.fixture
def ignore_index_data():
    predictions = np.array([[[0.2, 0.8, 0.6, 0.05, 0.78], [0.6, 0.4, 0.89, 0.05, 0.78]]])
    labels = np.array([[1, -100]])  # Only first token is valid
    return predictions, labels


@pytest.fixture
def all_wrong_data():
    predictions = np.array([[[0.9, 0.1, 0.89, 0.05, 0.78], [0.2, 0.8, 0.89, 0.05, 0.78]]])
    labels = np.array([[1, 0]])
    return predictions, labels


def test_perfect_prediction(perfect_prediction_data):
    predictions, labels = perfect_prediction_data
    metrics = compute_metrics(predictions, labels)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision_micro"] == 1.0
    assert metrics["recall_micro"] == 1.0
    assert metrics["f1_micro"] == 1.0
    assert metrics["precision_macro"] == 1.0
    assert metrics["recall_macro"] == 1.0
    assert metrics["f1_macro"] == 1.0


def test_partial_prediction(partial_prediction_data):
    predictions, labels = partial_prediction_data
    metrics = compute_metrics(predictions, labels)

    assert metrics["accuracy"] == 0.5
    assert round(metrics["precision_micro"], 2) == 0.5
    assert round(metrics["recall_micro"], 2) == 0.5
    assert round(metrics["f1_micro"], 2) == 0.5
    assert round(metrics["precision_macro"], 2) == 0.25
    assert round(metrics["recall_macro"], 2) == 0.5
    assert round(metrics["f1_macro"], 2) == 0.33



def test_with_ignore_index(ignore_index_data):
    predictions, labels = ignore_index_data
    metrics = compute_metrics(predictions, labels)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision_micro"] == 1.0
    assert metrics["recall_micro"] == 1.0
    assert metrics["f1_micro"] == 1.0
    assert round(metrics["precision_macro"], 2) == 1.0
    assert round(metrics["recall_macro"], 2) == 1.0
    assert round(metrics["f1_macro"], 2) == 1.0

def test_all_wrong_prediction(all_wrong_data):
    predictions, labels = all_wrong_data
    metrics = compute_metrics(predictions, labels)

    assert metrics["accuracy"] == 0.0
    assert metrics["precision_micro"] == 0.0
    assert metrics["recall_micro"] == 0.0
    assert metrics["f1_micro"] == 0.0
    assert round(metrics["precision_macro"], 2) == 0.0
    assert round(metrics["recall_macro"], 2) == 0.0
    assert round(metrics["f1_macro"], 2) == 0.0
