import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision import models

torch.cuda.empty_cache()

# Define ResNet-based model
class ResNetModel(nn.Module):
    def __init__(self, model_name="resnet18", num_classes=3, pretrained=True):
        super(ResNetModel, self).__init__()
        if model_name == "resnet18":
            self.model = models.resnet18(pretrained=pretrained)
        elif model_name == "resnet50":
            self.model = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError("Invalid model name. Use 'resnet18'.")
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# Evaluate global model
def evaluate_model(model, dataloader, device):
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total, total_loss / len(dataloader)
