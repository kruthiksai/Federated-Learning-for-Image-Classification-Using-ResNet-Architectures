from federated_averaging import federated_avg
from federated_sgd import federated_sgd
from federated_dyn import fed_dyn
from federated_scaffold import fed_scaffold

from security import encrypt_tensor, decrypt_tensor
from resnet_model import ResNetModel, evaluate_model

import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
import math

import torch

class FederatedLearning:
    def __init__(self, global_model1, global_model2, dataset_name, distribution, client_loaders, test_loader, rounds, local_epochs, device, alpha=0.1, lr=0.01):
        self.global_model1 = global_model1
        self.global_model2 = global_model2
        self.client_loaders = client_loaders
        self.test_loader = test_loader
        self.rounds = rounds
        self.local_epochs = local_epochs
        self.device = device
        self.alpha = alpha
        self.lr = lr
        self.algo_name = None
        self.dataset_name = dataset_name
        self.distribution = distribution

        # To store results for visualization
        self.accuracy_history_18 = None
        self.loss_history_18 = None
        self.weight_history_18 = None

        self.accuracy_history_50 = None
        self.loss_history_50 = None
        self.weight_history_50 = None

        self.client_weights_18 = None
        self.client_weights_50 = None
        
        # Flag to ensure run method is called before visualization
        self.is_run_executed = False

    def run(self, algo_name):
        self.algo_name = algo_name
        algo_mapping = {
            'fedavg': federated_avg,
            'fedsgd': federated_sgd,
            'feddyn': fed_dyn,
            'scaffold': fed_scaffold
        }

        if algo_name not in algo_mapping:
            raise ValueError(f"Unsupported algorithm: {algo_name}")

        algo_function = algo_mapping[algo_name]

        # Common parameters for all algorithms
        common_params = {
            'global_model': None,  # To be set dynamically
            'client_loaders': self.client_loaders,
            'test_loader' : self.test_loader,
            'rounds': self.rounds,
            'local_epochs': self.local_epochs,
            'device': self.device,
            'lr': self.lr,
        }

        self.global_model1.to(self.device)
        self.global_model2.to(self.device)
    
        # Additional parameters for specific algorithms
        additional_params = {}
        if algo_name == 'feddyn':
            additional_params = {'alpha': self.alpha}

        # Run the algorithm for ResNet-18
        print(f"\nRunning {algo_name} for ResNet-18")
        common_params['global_model'] = self.global_model1
        self.accuracy_history_18, self.loss_history_18, self.weight_history_18 = algo_function(**common_params, **additional_params)

        # Run the algorithm for ResNet-50
        print(f"\nRunning {algo_name} for ResNet-50")
        common_params['global_model'] = self.global_model2
        self.accuracy_history_50, self.loss_history_50, self.weight_history_50 = algo_function(**common_params, **additional_params)

        # Mark that the run method has been executed
        self.is_run_executed = True

    def _ensure_run_executed(self):
        if not self.is_run_executed:
            raise RuntimeError("You must call the `run` method at least once before visualizing results.")

    def plot_accuracy_comparison(self):
        self._ensure_run_executed()

        rounds = range(1, self.rounds + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(rounds, self.accuracy_history_18, marker='o', label="ResNet-18 Accuracy")
        plt.plot(rounds, self.accuracy_history_50, marker='^', label="ResNet-50 Accuracy")
        plt.xlabel("Rounds")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy Comparison Across Rounds ({self.algo_name.upper()})")
        plt.legend()
        plt.grid(True)
        plt.savefig(f'artifacts/{self.algo_name}_{self.dataset_name}_{self.distribution}_Accuracy_Graph.png')

    def plot_loss_visualization(self):
        self._ensure_run_executed()

        rounds = range(1, self.rounds + 1)

        loss_18_combined = [round_loss if not math.isnan(round_loss) else 0 for round_loss in self.loss_history_18]
        loss_50_combined = [round_loss if not math.isnan(round_loss) else 0 for round_loss in self.loss_history_50]
        
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, loss_18_combined, marker='o', label="ResNet-18 Loss")
        plt.plot(rounds, loss_50_combined, marker='^', label="ResNet-50 Loss")
        plt.xlabel("Rounds")
        plt.ylabel("Loss")
        plt.title(f"Loss Comparison Across Rounds ({self.algo_name.upper()})")
        plt.legend()
        plt.grid(True)
        plt.savefig(f'artifacts/{self.algo_name}_{self.dataset_name}_{self.distribution}_Loss_Graph.png')
        

    def evaluate_with_metrics(self, global_model, dataloader, label, classes):
        predictions, ground_truths = self.infer(global_model, dataloader)

        cm = confusion_matrix(ground_truths, predictions, labels=list(range(len(classes))))
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title(f"Confusion Matrix: {label}")
        plt.savefig(f"artifacts/{self.algo_name}_{self.dataset_name}_{self.distribution}_{label.split(' ')[1]}_Confusion_Matrix.png")

        metrics_report = classification_report(ground_truths, predictions, target_names=classes, output_dict=True)
        print(f"Classification Report: {label}")
        print(classification_report(ground_truths, predictions, target_names=classes))

        return cm, metrics_report

    def infer(self, model, dataloader):
        model.eval()
        predictions, labels = [], []
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().tolist())
                labels.extend(targets.tolist())
        return predictions, labels

    def perform_inference_with_images(self, global_model, label, dataloader, classes):
        print(f"\nPerforming Inference: {label}")
        global_model.eval()

        images_shown = 0
        predictions, ground_truths = [], []
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = global_model(inputs)
                _, predicted = torch.max(outputs, 1)

                predictions.extend(predicted.cpu().tolist())
                ground_truths.extend(targets.cpu().tolist())

                if images_shown == 0:
                    inputs = inputs.cpu()
                    plt.figure(figsize=(12, 12))
                    for i in range(min(len(inputs), 10)):
                        plt.subplot(5, 5, i + 1)
                        plt.imshow(inputs[i].permute(1, 2, 0).squeeze(), cmap="gray")
                        plt.title(f"Pred: {classes[predicted[i]]}\nTrue: {classes[targets[i]]}", fontsize=8)
                        plt.axis("off")
                    plt.suptitle(f"Inference Results: {label}", fontsize=16)
                    plt.savefig(f"artifacts/{self.algo_name}_{self.dataset_name}_{self.distribution}_{label.split(' ')[1]}_Inference.png", bbox_inches="tight")

                    images_shown = 1

        for i in range(10):
            print(f"Prediction: {classes[predictions[i]]}, Ground Truth: {classes[ground_truths[i]]}")