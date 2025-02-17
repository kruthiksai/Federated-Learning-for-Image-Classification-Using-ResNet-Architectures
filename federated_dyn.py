from security import encrypt_tensor, decrypt_tensor
from resnet_model import ResNetModel, evaluate_model
from torch import nn, optim
import copy
import torch

encryption_key = b'project_group14a'

def fed_dyn(global_model, client_loaders, test_loader, rounds, local_epochs, device, alpha, lr=0.01):
    n_clients = len(client_loaders)
    accuracy_history, loss_history, weight_history = [], [], []
    client_accuracies_history = []  # Store client accuracies for each round

    # Initialize global control variables for FedDyn
    control = {key: torch.zeros_like(value, dtype=torch.float32).to(device) 
               for key, value in global_model.state_dict().items()}

    for round_num in range(1, rounds + 1):
        print(f"\n--- FedDyn: Round {round_num} ---")
        encrypted_client_weights = []
        global_state_dict = global_model.state_dict()
        round_client_accuracies = []  # Store accuracies for all clients in this round

        for client_idx, client_loader in enumerate(client_loaders):
            # Initialize client model
            client_model = copy.deepcopy(global_model).to(device)
            client_optimizer = optim.SGD(client_model.parameters(), lr)
            client_model.train()

            # Train locally for multiple epochs
            for epoch in range(local_epochs):
                epoch_loss = 0.0
                for inputs, labels in client_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    client_optimizer.zero_grad()
                    outputs = client_model(inputs)
                    loss = nn.CrossEntropyLoss()(outputs, labels)

                    total_loss = loss
                    total_loss.backward()
                    client_optimizer.step()

                    epoch_loss += total_loss.item()
                print(f"Client {client_idx + 1}, Epoch {epoch + 1}/{local_epochs}, Loss: {epoch_loss:.4f}")

            # Evaluate client model after local training
            client_accuracy, client_avg_loss = evaluate_model(client_model, client_loader, device)
            print(f"Client {client_idx + 1} - Accuracy: {client_accuracy:.2f}, Loss: {client_avg_loss:.4f}")
            round_client_accuracies.append(client_accuracy)

            # Encrypt client weights
            encrypted_weights = {
                key: encrypt_tensor(param.clone().detach().float(), encryption_key)
                for key, param in client_model.state_dict().items()
            }
            encrypted_client_weights.append(encrypted_weights)

        # Decrypt and aggregate client weights to update global model
        aggregated_weights = {}
        for key in global_state_dict.keys():
            decrypted_weights = [
                decrypt_tensor(encrypted_client_weights[client_idx][key], 
                               shape=global_state_dict[key].shape, 
                               dtype=torch.float32, 
                               key=encryption_key).to(device)
                for client_idx in range(len(encrypted_client_weights))
            ]
            aggregated_weights[key] = torch.mean(torch.stack(decrypted_weights), dim=0)

        # Update control variables
        for key in control.keys():
            control[key] -= alpha * (aggregated_weights[key] - global_state_dict[key].float()) * len(encrypted_client_weights) / n_clients

        # Update global model weights with control regularization
        for key in aggregated_weights.keys():
            aggregated_weights[key] -= control[key] / alpha

        global_model.load_state_dict(aggregated_weights)

        # Store client accuracies for the round
        client_accuracies_history.append(round_client_accuracies)

        # Evaluate global model
        accuracy, avg_loss = evaluate_model(global_model, test_loader, device)
        accuracy_history.append(accuracy)
        loss_history.append(avg_loss)
        weight_history.append(copy.deepcopy(global_model.state_dict()))

        print(f"Global Model, Round {round_num}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}")

    return accuracy_history, loss_history, weight_history