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



## 📁 Project Folder Structure
```
Federated-Learning/
├── artifacts/                                 # Stores generated accuracy/loss graphs and confusion matrices
│   ├── New folder/                            # Results for CUSTOM dataset
│   │   ├── fedavg_CUSTOM_IID_Accuracy_Graph_4clients.png
│   │   ├── fedavg_CUSTOM_IID_Loss_Graph_4clients.png
│   │   ├── fedavg_CUSTOM_IID_ResNet-18_Confusion_Matrix.png
│   │   ├── fedavg_CUSTOM_IID_ResNet-18_Inference.png
│   │   ├── fedavg_CUSTOM_IID_ResNet-50_Confusion_Matrix.png
│   │   └── fedavg_CUSTOM_IID_ResNet-50_Inference.png
│   ├── artifacts_rounds_10_clients_5_epochs_5_IID/
│   ├── artifacts_rounds_20_clients_3_epochs_5_IID/
│   ├── artifacts_rounds_20_clients_5_epochs_5_IID/
│   └── artifacts_rounds_25_clients_3_nonIID/
│
├── logs_rounds_10_clients_5_epochs_3_IID/     # Logs per round (training progress)
├── logs_rounds_20_clients_3_epochs_5_IID/
├── logs_rounds_20_clients_5_epochs_5_IID/
├── logs_rounds_25_clients_3_nonIID/
│
├── data/                                      # Folder containing training/test data
│
├── .gitignore
├── data_loader.py                             # Loads and partitions datasets
├── data_split.py                              # Custom data splitting utility
├── federated_averaging.py                     # Implementation of FedAvg
├── federated_dyn.py                           # Implementation of FedDyn
├── federated_learning.py                      # Main federated learning loop logic
├── federated_scaffold.py                      # Implementation of SCAFFOLD
├── federated_sgd.py                           # Implementation of FedSGD
├── resnet_model.py                            # ResNet-18 and ResNet-50 models + evaluation
├── security.py                                # Placeholder for any future encryption logic
├── main.py                                    # Entry point to run the training script
└── README.md                                  # Project documentation
```

## 📊 Results

### ✅ Accuracy Comparison Plot (ResNet-18 vs ResNet-50)
![Accuracy Comparison](artifacts/New%20folder/fedavg_CUSTOM_IID_Accuracy_Graph_4clients.png)

### ✅ Loss Visualization per Round
![Loss Visualization](artifacts/New%20folder/fedavg_CUSTOM_IID_Loss_Graph_4clients.png)

### ✅ Confusion Matrix – ResNet-18
![Confusion Matrix ResNet-18](artifacts/New%20folder/fedavg_CUSTOM_IID_ResNet-18_Confusion_Matrix.png)

### ✅ Confusion Matrix – ResNet-50
![Confusion Matrix ResNet-50](artifacts/New%20folder/fedavg_CUSTOM_IID_ResNet-50_Confusion_Matrix.png)

### ✅ Inference Samples – ResNet-18
![Inference ResNet-18](artifacts/New%20folder/fedavg_CUSTOM_IID_ResNet-18_Inference.png)

### ✅ Inference Samples – ResNet-50
![Inference ResNet-50](artifacts/New%20folder/fedavg_CUSTOM_IID_ResNet-50_Inference.png)



