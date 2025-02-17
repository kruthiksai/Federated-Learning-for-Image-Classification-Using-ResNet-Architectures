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

## Outputs

The simulation generates several outputs that help evaluate the performance of the federated learning models. Here is a detailed breakdown of the output files stored in the `artifacts` folder:

### Confusion Matrices
- **ResNet-18 Confusion Matrix:** Shows the performance of the ResNet-18 model in classifying images into categories such as COVID, Normal, and Viral Pneumonia. This matrix helps visualize the accuracy of predictions against true labels.
<img src="https://github.com/kruthiksai/Federated-Learning-for-Image-Classification-Using-ResNet-Architectures/blob/main/artifacts_rounds_20_epoch_5_clients_5_IID/fedavg_FMNIST_IID_ResNet-18_Confusion_Matrix.png" width="25%" alt="fedavg_FMNIST_IID_ResNet-18 Confusion Matrix">

- **ResNet-50 Confusion Matrix:** Similar to the ResNet-18 matrix but for the ResNet-50 model, providing insight into its classification effectiveness.
<img src="https://github.com/kruthiksai/Federated-Learning-for-Image-Classification-Using-ResNet-Architectures/blob/main/artifacts_rounds_20_epoch_5_clients_5_IID/fedavg_FMNIST_IID_ResNet-50_Confusion_Matrix.png" width="25%" alt="fedavg_FMNIST_IID_ResNet-50 Confusion Matrix">

### Accuracy and Loss Graphs
- **Accuracy Comparison Across Rounds:** This graph plots the accuracy of the ResNet-18 and ResNet-50 models across different rounds of federated learning. It is crucial for understanding how model accuracy evolves with each communication round between clients.
- **Loss Comparison Across Rounds:** Illustrates the change in loss for both ResNet models across federated learning rounds, highlighting how model performance improves or degrades over time.

### Inference Results
- **ResNet-50 Inference Results:** Displays a series of X-ray images alongside their predicted and true labels, demonstrating the ResNet-50 model’s ability to classify medical images under real-world conditions.
- **ResNet-18 Inference Results:** Similar to the ResNet-50 results but showcasing the inference capabilities of the ResNet-18 model.

These outputs collectively offer a comprehensive view of the model's performance across different metrics and practical scenarios, providing valuable insights into the effectiveness of federated learning strategies employed.
