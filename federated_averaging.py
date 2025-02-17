from security import encrypt_tensor, decrypt_tensor
from resnet_model import ResNetModel, evaluate_model
from torch import nn, optim
import copy
import torch

encryption_key = b'project_group14a'

def federated_avg(global_model, client_loaders, test_loader, rounds, local_epochs, device, lr=0.01):
    accuracy_history, loss_history, weight_history = [], [], []
    client_accuracies_history = []  # To store accuracies of each client for each round
    client_models = [copy.deepcopy(global_model).to(device) for _ in client_loaders]  # Persistent client models

    for round_num in range(1, rounds + 1):
        print(f"\n--- FedAvg: Round {round_num} ---")
        client_weights = []
        client_accuracies = []  # Accuracies for this round

        for client_idx, (client_model, client_loader) in enumerate(zip(client_models, client_loaders)):
            optimizer = optim.SGD(client_model.parameters(), lr)
            #optimizer = torch.optim.Adam(client_model.parameters(), lr)
            client_model.train()

            for epoch in range(local_epochs):
                epoch_loss = 0.0
                for inputs, labels in client_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = client_model(inputs)
                    loss = nn.CrossEntropyLoss()(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                print(f"Client {client_idx + 1}, Epoch {epoch + 1}/{local_epochs}, Loss: {epoch_loss:.4f}")

            # Evaluate the client model and store accuracy
            client_accuracy, _ = evaluate_model(client_model, test_loader, device)
            client_accuracies.append(client_accuracy)
            print(f"Client {client_idx + 1} Accuracy (Round {round_num}): {client_accuracy:.4f}")


            # Encrypt client weights
            encrypted_weights = {
                key: encrypt_tensor(value.clone(), encryption_key)
                for key, value in client_model.state_dict().items()
            }
            client_weights.append(encrypted_weights)

        # Store client accuracies for this round
        client_accuracies_history.append(client_accuracies)

        # Aggregate encrypted weights and update global model
        global_state_dict = global_model.state_dict()
        for key in global_state_dict:
            decrypted_weights = [
                decrypt_tensor(client[key], global_state_dict[key].shape, global_state_dict[key].dtype, encryption_key)
                for client in client_weights
            ]
            global_state_dict[key] = sum(decrypted_weights) / len(decrypted_weights)
        global_model.load_state_dict(global_state_dict)
        weight_history.append(copy.deepcopy(global_model.state_dict()))

        # Evaluate global model
        accuracy, avg_loss = evaluate_model(global_model, test_loader, device)
        accuracy_history.append(accuracy)
        loss_history.append(avg_loss)
        print(f"Global Model, Round {round_num}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}")

    return accuracy_history, loss_history, weight_history