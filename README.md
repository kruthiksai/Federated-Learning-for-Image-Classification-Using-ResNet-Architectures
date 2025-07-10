# Federated Learning Image Classification

This project implements Federated Learning algorithms (e.g., FedAvg, FedSGD, FedDyn, SCAFFOLD) to perform image classification using ResNet models (ResNet-18 and ResNet-50). It supports multiple datasets and customizable training configurations, including IID and non-IID data distributions.

## 📌 Features

- Supports **Federated Learning** algorithms: `FedAvg`, `FedSGD`, `FedDyn`, `SCAFFOLD`
- Trains on multiple datasets: `FMNIST`, `CIFAR`, and `CUSTOM`
- Two deep learning models: `ResNet-18` and `ResNet-50`
- Visualization of accuracy and loss across rounds
- Confusion matrix and classification report
- Inference on test images

---

## 🚀 How to Run

Make sure to install the required dependencies:

```bash
pip install -r requirements.txt

```
Run the project using the following command:

python main.py --dataset_name=CUSTOM --algo_name=fedavg --rounds=1 --num_clients=1 --distribution=IID --local_epochs=25

## ⚙️ Available Command-Line Parameters
---------------------------------------------------------------------------------------------
| Parameter        | Description                                                            |
| ---------------- | ---------------------------------------------------------------------- |
| `--algo_name`    | Federated Learning algorithm: `fedavg`, `fedsgd`, `feddyn`, `scaffold` |
| `--dataset_name` | Dataset to use: `FMNIST`, `CIFAR`, `CUSTOM`                            |
| `--distribution` | Data distribution: `IID`, `non-IID`                                    |
| `--num_clients`  | Number of participating clients (e.g., 5, 10)                          |
| `--rounds`       | Number of federated training rounds                                    |
| `--local_epochs` | Number of local epochs per client                                      |
---------------------------------------------------------------------------------------------



