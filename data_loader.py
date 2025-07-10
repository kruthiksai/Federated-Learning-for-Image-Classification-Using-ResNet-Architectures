from data_split import create_iid_partitions, create_non_iid_partitions
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset


import numpy as np

encryption_key = b'project_group14a'


def data_loader_function(dataset_name, num_clients, data_type = 'IID', num_classes_per_client = 3):

    if dataset_name == 'CUSTOM':
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize images to a consistent size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization parameters
        ])
        custom_dataset_path = 'C:/Users/saikr/Downloads/chest_xray'  # Adjust path as necessary
        dataset = ImageFolder(root=custom_dataset_path + '/train', transform=transform)
        test_dataset = ImageFolder(root=custom_dataset_path + '/test', transform=transform)
        classes = dataset.classes



    elif dataset_name == 'CIFAR':
        
        # Define transformations for CIFAR-10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalize each channel to [-1, 1]
        ])
        
        # Load CIFAR-10 dataset
        dataset = datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )


    else:

        # Define transformations for Fashion-MNIST
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # For ResNet compatibility
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to [-1, 1]
        ])
        
        # Load Fashion-MNIST dataset
        dataset = datasets.FashionMNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = datasets.FashionMNIST(
            root='./data', train=False, download=True, transform=transform
        )


    classes = dataset.classes
    print(classes)
    print(len(dataset))
    #dataset = Subset(dataset, np.random.choice(len(dataset), 1000, replace=False))
    #test_dataset = Subset(test_dataset, np.random.choice(len(test_dataset), 100, replace=False))

    if data_type == 'IID':
        client_datasets = create_iid_partitions(dataset, num_clients)
        client_loaders = [
            DataLoader(client_dataset, batch_size=32, shuffle=True, num_workers=16, pin_memory=True)
            for client_dataset in client_datasets
        ]
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=16, pin_memory=True)

    else:
        client_datasets = create_non_iid_partitions(dataset,num_clients, num_classes_per_client)
        client_loaders = [
            DataLoader(client_dataset, batch_size=32, shuffle=True, num_workers=16, pin_memory=True)
            for client_dataset in client_datasets
        ]
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=16, pin_memory=True)

    return [client_loaders, test_loader, classes]
    
        