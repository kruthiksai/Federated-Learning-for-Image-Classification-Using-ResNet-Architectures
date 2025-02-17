import argparse
import sys
import torch
import copy
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
import math

import warnings
warnings.filterwarnings("ignore")

from data_loader import data_loader_function
from federated_learning import FederatedLearning
from resnet_model import ResNetModel, evaluate_model
torch.manual_seed(42)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate a random encryption key (shared among all clients and server)
encryption_key = b'project_group14a'


def main():
    # Define valid values
    valid_algos = {"fedavg", "fedsgd", "feddyn", "scaffold"}
    valid_datasets = {"FMNIST", "CIFAR", "CUSTOM"}
    valid_distributions = {"IID", "non-IID"}

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run Federated Learning Simulation.")
    parser.add_argument(
        "--algo_name", required=True, type=str, help="Algorithm name (fedavg, fedsgd, feddyn, scaffold)"
    )
    parser.add_argument(
        "--dataset_name", required=True, type=str, help="Dataset name (FMNIST, CIFAR)"
    )
    parser.add_argument(
        "--distribution", required=True, type=str, help="Data distribution (IID, non-IID)"
    )
    parser.add_argument("--num_clients", type=int, required=True,
                        help="Number of clients (e.g., 5, 10, etc.)")
    parser.add_argument("--rounds", type=int, required=True,
                        help="Number of communication rounds (e.g., 10, 50, etc.)")
    parser.add_argument("--local_epochs", type=int, required=True,
                        help="Number of local epochs per client (e.g., 1, 5, etc.)")

    # Parse arguments
    args = parser.parse_args()

    # Validate arguments
    if args.algo_name not in valid_algos:
        print(f"Error: Invalid algo_name '{args.algo_name}'. Allowed values: {', '.join(valid_algos)}")
        sys.exit(1)
    if args.dataset_name not in valid_datasets:
        print(f"Error: Invalid dataset_name '{args.dataset_name}'. Allowed values: {', '.join(valid_datasets)}")
        sys.exit(1)
    if args.distribution not in valid_distributions:
        print(f"Error: Invalid distribution '{args.distribution}'. Allowed values: {', '.join(valid_distributions)}")
        sys.exit(1)
    if args.num_clients <= 0:
        print("Error: num_clients must be greater than 0.")
        sys.exit(1)
    if args.rounds <= 0:
        print("Error: rounds must be greater than 0.")
        sys.exit(1)
    if args.local_epochs <= 0:
        print("Error: local_epochs must be greater than 0.")
        sys.exit(1)

    # Store arguments in variables
    algo_name = args.algo_name
    dataset_name = args.dataset_name
    distribution = args.distribution
    num_clients = args.num_clients
    rounds = args.rounds
    local_epochs = args.local_epochs

    # Display the parsed and validated arguments
    print("Federated Learning Configuration:")
    print(f"  Algorithm: {algo_name}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Distribution: {distribution}")
    print(f"  Number of Clients: {num_clients}")
    print(f"  Rounds: {rounds}")
    print(f"  Local Epochs: {local_epochs}")

    number_of_classes_per_client = 3 // num_clients
    client_loaders, test_loader, classes = data_loader_function(dataset_name, num_clients,distribution, number_of_classes_per_client)
    federated_learning = FederatedLearning(
        global_model1=ResNetModel(model_name="resnet18", num_classes=3),
        global_model2=ResNetModel(model_name="resnet50", num_classes=3),
        dataset_name=dataset_name,
        distribution = distribution,
        client_loaders=client_loaders,
        test_loader = test_loader,
        rounds=rounds,  # Set appropriate number of rounds
        local_epochs=local_epochs,  # Set appropriate local epochs
        device=device,
        alpha=0.1,
    )
    federated_learning.run(algo_name=algo_name)
    federated_learning.plot_accuracy_comparison()
    federated_learning.plot_loss_visualization()

    # Evaluate ResNet-18
    cm_18, metrics_18 = federated_learning.evaluate_with_metrics(
        global_model=federated_learning.global_model1,
        dataloader=test_loader,
        label=f"{algo_name} ResNet-18",
        classes=classes
    )

    # Evaluate ResNet-50
    cm_50, metrics_50 = federated_learning.evaluate_with_metrics(
        global_model=federated_learning.global_model2,
        dataloader=test_loader,
        label=f"{algo_name} ResNet-50",
        classes=classes
    )

    # Perform inference for ResNet-18
    federated_learning.perform_inference_with_images(
        global_model=federated_learning.global_model1,
        label=f"{algo_name} ResNet-18",
        dataloader=test_loader,
        classes=classes
    )

    # Perform inference for ResNet-50
    federated_learning.perform_inference_with_images(
        global_model=federated_learning.global_model2,
        label=f"{algo_name} ResNet-50",
        dataloader=test_loader,
        classes=classes
    )



if __name__ == "__main__":
    #sys.stdout = open("output.log", "w")
    main()
    #sys.stdout.close()