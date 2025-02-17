from security import encrypt_tensor, decrypt_tensor
from resnet_model import ResNetModel, evaluate_model
from torch import nn, optim
import copy
import torch

encryption_key = b'project_group14a'

def federated_sgd(global_model, client_loaders, test_loader, rounds, local_epochs, device, lr=0.01):

    accuracy_history, loss_history, weight_history = [], [], []
    global_model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    batch_limiter = 10
    
    for round_num in range(1, rounds + 1):
        print(f"\n--- FedSGD: Round {round_num} ---")
        global_gradients = None
        total_samples = sum(len(loader.dataset) for loader in client_loaders)
        encrypted_gradients = []

        for client_idx, client_loader in enumerate(client_loaders):
            # Create a fresh copy of the global model for each client
            client_model = copy.deepcopy(global_model).to(device)
            client_model.train()
            batch = 0
            # Compute gradients over the entire dataset
            for inputs, labels in client_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = client_model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()

                # Collect encrypted and scaled gradients for each batch
                batch_grad = {
                    name: encrypt_tensor(
                        param.grad.clone().detach() * (len(client_loader.dataset) / total_samples),
                        encryption_key
                    )
                    for name, param in client_model.named_parameters() if param.grad is not None
                }
                encrypted_gradients.append(batch_grad)
                batch = batch + 1
                if batch == batch_limiter:
                    break

            # Evaluate the client model and store accuracy
            client_accuracy = evaluate_model(client_model, client_loader, device)
            print(f"Client {client_idx + 1} Accuracy (Round {round_num}): {client_accuracy:.4f}")
        
        # Aggregate global gradients
        global_gradients = {
            name: sum(
                decrypt_tensor(
                    grad[name],
                    global_model.state_dict()[name].shape,
                    global_model.state_dict()[name].dtype,
                    encryption_key
                ).to(device)
                for grad in encrypted_gradients
            ) / len(encrypted_gradients)
            for name in global_model.state_dict().keys() if name in encrypted_gradients[0]
        }

        # Update the global model with aggregated gradients
        with torch.no_grad():
            for param_name, grad in global_gradients.items():
                global_model.state_dict()[param_name] -= lr * grad

        # Save model state for tracking
        weight_history.append(copy.deepcopy(global_model.state_dict()))

        # Evaluate global model
        accuracy, avg_loss = evaluate_model(global_model, test_loader, device)
        accuracy_history.append(accuracy)
        loss_history.append(avg_loss)

        print(f"Global Model, Round {round_num}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    return accuracy_history, loss_history, weight_history