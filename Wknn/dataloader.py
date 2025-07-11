# -*- coding: utf-8 -*-
# @author: H. T. Duong V.

import numpy as np
from typing import Tuple
import random
import scipy as scipy

random.seed(42) # For reproducibility

def generate_train_test(desc_set: np.ndarray, label_set: np.ndarray, n_per_class: int) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Génère des ensembles d'entraînement et de test avec un nombre donné d'exemples par classe pour l'entraînement.

    Args:
        desc_set (ndarray): Données (descriptions).
        label_set (ndarray): Étiquettes correspondantes.
        n_per_class (int): Nombre d'exemples par classe pour l'entraînement.

    Returns:
        tuple: ((train_data, train_labels), (test_data, test_labels))
    """
    unique_labels = np.unique(label_set)
    
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    for label in unique_labels:
        class_indices = np.where(label_set == label)[0]
        
        if len(class_indices) < n_per_class:
            raise ValueError(f"Pas assez d'exemples pour la classe {label}")
        
        selected_indices = random.sample(class_indices.tolist(), n_per_class)
        selected_set = set(selected_indices)

        for idx in class_indices:
            if idx in selected_set:
                train_data.append(desc_set[idx])
                train_labels.append(label)
            else:
                test_data.append(desc_set[idx])
                test_labels.append(label)

    return (
        (np.array(train_data), np.array(train_labels)),
        (np.array(test_data), np.array(test_labels))
    )


def generate_uniform_dataset_multiclass(p: int, n_per_class: int, classes: list, binf: float = -1, bsup: float = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a uniformly distributed dataset for multiple classes.

    :param p: Number of features
    :param n_per_class: Number of samples per class
    :param classes: Iterable of class labels (can include -9 to 9, including 0)
    :param binf: Lower bound for uniform distribution
    :param bsup: Upper bound for uniform distribution
    :return: Tuple (data_desc, data_labels)
    """
    data_desc = []
    data_labels = []

    for label in classes:
        samples = np.random.uniform(binf, bsup, (n_per_class, p))
        data_desc.append(samples)
        data_labels.extend([label] * n_per_class)

    data_desc = np.vstack(data_desc)
    data_labels = np.array(data_labels)

    return data_desc, data_labels


    

def generate_gaussian_dataset(centers: np.ndarray, sigmas: np.ndarray, labels: np.ndarray, nb_points_per_class: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a dataset from multiple Gaussian distributions.
    
    :param centers: List of mean vectors (one per class)
    :param sigmas: List of covariance matrices (one per class)
    :param labels: List of class labels (can be negative, zero, or positive)
    :param nb_points_per_class: Number of points to generate per class
    :return: Tuple (data_desc, data_labels)
    """

    # Check dim
    if len(centers) != len(sigmas):
        raise ValueError("centers, sigmas, and labels must have the same length")
    if len(centers) == 0:
        raise ValueError("centers, sigmas, and labels cannot be empty")
    if len(centers[0]) != len(sigmas[0]):
        raise ValueError("centers and sigmas must have the same dimension")
        
    # Convert to numpy arrays
    centers = np.array(centers)
    sigmas = np.array(sigmas)
    labels = np.array(labels)

    # Check if all covariance matrices are square and symmetric
    for sigma in sigmas:
        if sigma.shape[0] != sigma.shape[1]:
            raise ValueError("Covariance matrices must be square")
        if not np.allclose(sigma, sigma.T):
            raise ValueError("Covariance matrices must be symmetric")
        
    data = []
    data_labels = []
    
    for center, sigma, label in zip(centers, sigmas, labels):
        points = np.random.multivariate_normal(center, sigma, nb_points_per_class)
        data.append(points)
        data_labels += [label] * nb_points_per_class
    
    data = np.vstack(data)
    data_labels = np.array(data_labels)
    
    return data, data_labels




def dynamic_generate_positive_gaussian_data(dim: int, nb_classes: int, nb_points_per_class: int, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates all-positive multivariate Gaussian data.

    :param dim: Number of dimensions
    :param nb_classes: Number of Gaussian classes
    :param nb_points_per_class: Number of samples per class
    :param seed: Optional random seed for reproducibility
    :return: Tuple (data, labels)
    """
    if seed is not None:
        np.random.seed(seed)

    # Create centers: spaced out to avoid overlaps
    centers = [np.full(dim, 5.0 * (i + 1)) for i in range(nb_classes)]

    # Create small positive definite covariance matrices
    sigmas = []
    for _ in range(nb_classes):
        A = np.random.rand(dim, dim) * 0.2
        sigma = np.dot(A, A.T) + np.eye(dim) * 0.5
        sigmas.append(sigma)

    labels = list(range(nb_classes))

    # Generate data using your existing function
    data, data_labels = generate_gaussian_dataset(centers, sigmas, labels, nb_points_per_class)

    # Keep only positive samples
    positive_mask = np.all(data > 0, axis=1)
    data = data[positive_mask]
    data_labels = data_labels[positive_mask]

    return data, data_labels