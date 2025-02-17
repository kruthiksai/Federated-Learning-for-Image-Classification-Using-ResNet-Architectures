# Federated Learning for Image Classification Using ResNet Architectures

This project simulates a federated learning environment for image classification using ResNet architectures. It explores different federated learning algorithms and their impact on model performance across various data distributions and datasets.

## Requirements

Ensure you have Python 3.x installed along with the following packages:
- `torch`
- `matplotlib`
- `sklearn`
- `seaborn`
- `numpy`

## Configuration and Usage

To run the simulation, you need to specify several parameters that configure the federated learning environment and the neural network models. Use the following command to execute the simulation:

```bash
python main.py --dataset_name=CUSTOM --algo_name=fedavg --rounds=2 --num_clients=2 --distribution=IID --local_epochs=5
```

## Inputs
The script accepts the following command-line arguments to configure the federated learning simulation:

 - --algo_name: Specifies the federated learning algorithm. Valid options are `fedavg`, `fedsgd`, `feddyn`, `scaffold`.
 - --dataset_name: Name of the dataset to be used. Supported datasets are `FMNIST`, `CIFAR`, `CUSTOM`.
- --distribution: Type of data distribution among clients. Options include `IID` and `non-IID`.
- --num_clients: The number of clients participating in the learning process.
- --rounds: The number of communication rounds in the federated learning process.
- --local_epochs: The number of epochs each client will train locally before averaging.
