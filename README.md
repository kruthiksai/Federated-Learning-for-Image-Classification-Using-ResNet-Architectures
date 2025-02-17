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
