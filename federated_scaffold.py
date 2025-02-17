from security import encrypt_tensor, decrypt_tensor
from resnet_model import ResNetModel, evaluate_model
from torch import nn, optim
import copy
import torch

encryption_key = b'project_group14a'

def fed_scaffold(global_model, client_loaders, test_loader, rounds, local_epochs, device, lr=0.01):

    global_model.to(device)
    accuracy_history, loss_history, weight_history = [], [], []

    # Initialize global control variate
    global_control = {name: torch.zeros_like(param, device=device) for name, param in global_model.named_parameters()}

    # Initialize client control variates
    client_controls = [
        {name: torch.zeros_like(param, device=device) for name, param in global_model.named_parameters()}
        for _ in client_loaders
    ]

    loss_fn = nn.CrossEntropyLoss()

    for round_num in range(1, rounds + 1):
        print(f"\n--- SCAFFOLD: Round {round_num} ---")
        client_updates = []

        for client_idx, (client_loader, client_control) in enumerate(zip(client_loaders, client_controls)):
            client_model = copy.deepcopy(global_model).to(device)
            client_model.train()
            optimizer = optim.SGD(client_model.parameters(), lr=lr)

            for epoch in range(1, local_epochs + 1):
                epoch_loss = 0.0
                for inputs, labels in client_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = client_model(inputs)
                    loss = loss_fn(outputs, labels)

                    # Compute the difference between global and client controls
                    control_diff = {
                        name: global_control[name] - client_control[name]
                        for name in global_control
                    }

                    # Apply control variate correction
                    for name, param in client_model.named_parameters():
                        if param.grad is not None:
                            loss += torch.sum(control_diff[name] * param)

                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                print(f"Client {client_idx + 1}, Epoch {epoch}/{local_epochs}, Loss: {epoch_loss:.4f}")

            # Compute client update
            client_update = {
                name: param.data - global_model.state_dict()[name]
                for name, param in client_model.state_dict().items()
            }
            client_updates.append(client_update)

            # Update client control variate
            client_controls[client_idx] = {
                name: client_control[name] + client_update[name] / (lr * local_epochs)
                for name in client_control
            }

        # Aggregate client updates
        avg_update = {
            name: sum(client_update[name] for client_update in client_updates) / len(client_updates)
            for name in global_model.state_dict()
        }

        # Update global model
        global_model.load_state_dict({
            name: param + avg_update[name]
            for name, param in global_model.state_dict().items()
        })

        # Update global control variate
        global_control = {
            name: global_control[name] + avg_update[name] / (lr * local_epochs)
            for name in global_control
        }

        # Evaluate global model
        accuracy, avg_loss = evaluate_model(global_model, test_loader, device)
        accuracy_history.append(accuracy)
        loss_history.append(avg_loss)
        weight_history.append(copy.deepcopy(global_model.state_dict()))
        
        print(f"Global Model, Round {round_num}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}")

    return accuracy_history, loss_history, weight_history