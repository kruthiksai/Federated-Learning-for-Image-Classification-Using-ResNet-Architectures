from torch.utils.data import Subset
import numpy as np

def create_iid_partitions(dataset, num_clients=5):
    """
    Simulate IID data partitioning across clients.
    """
    num_items = len(dataset) // num_clients  # Number of samples per client
    all_indices = np.arange(len(dataset))  # All dataset indices
    np.random.shuffle(all_indices)  # Shuffle the indices for randomness

    # Split indices equally among clients
    client_data_indices = [
        all_indices[i * num_items: (i + 1) * num_items] for i in range(num_clients)
    ]

    # Return datasets for each client as Subsets
    return [Subset(dataset, indices) for indices in client_data_indices]


def create_non_iid_partitions(dataset, num_clients=5, num_classes_per_client=3):
    """
    Simulate Non-IID data partitioning across clients.
    Args:
        dataset: PyTorch dataset object.
        num_clients: Number of clients.
        num_classes_per_client: Number of classes assigned to each client.
    Returns:
        List of Subsets, one for each client.
    """
    # Create a mapping of class indices to sample indices
    targets = np.array(dataset.targets)  # Targets (labels) for the dataset
    class_indices = {class_idx: np.where(targets == class_idx)[0] for class_idx in np.unique(targets)}

    # Shuffle the indices within each class for randomness
    for class_idx in class_indices:
        np.random.shuffle(class_indices[class_idx])

    # Assign specific classes to each client
    client_data_indices = [[] for _ in range(num_clients)]
    available_classes = list(class_indices.keys())
    np.random.shuffle(available_classes)  # Shuffle class assignments

    for client_idx in range(num_clients):
        # Assign `num_classes_per_client` classes to the current client
        assigned_classes = available_classes[client_idx * num_classes_per_client: (client_idx + 1) * num_classes_per_client]
        for class_idx in assigned_classes:
            # Split data for the current client
            client_data_indices[client_idx].extend(class_indices[class_idx][:len(class_indices[class_idx]) // num_clients])
            # Update the class_indices to remove the assigned samples
            class_indices[class_idx] = class_indices[class_idx][len(class_indices[class_idx]) // num_clients:]

    # Create Subsets for each client
    return [Subset(dataset, indices) for indices in client_data_indices]
