import os
import sys

# Thêm đường dẫn thư mục cha (project/) vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from libs.utils import preprocess_data, Trainer
from optimizations_algorithms.optimizers import SGD
from nn_components.losses import CrossEntropy
from sklearn.metrics import confusion_matrix
from softmaxregression import SoftmaxRegression



# Tải bộ dữ liệu MNIST từ PyTorch
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),  # Chuyển đổi hình ảnh thành Tensor
        transforms.Normalize((0.5,), (0.5,))  # Chuẩn hóa dữ liệu (mean=0.5, std=0.5)
    ])
    
    # Tải tập huấn luyện và kiểm tra
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Tạo DataLoader cho tập huấn luyện và kiểm tra
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)
    
    return trainloader, testloader

if __name__ == '__main__':
    # Tải dữ liệu MNIST từ PyTorch
    trainloader, testloader = load_data()

    # Chuẩn bị dữ liệu huấn luyện và kiểm tra
    images_train = []
    labels_train = []
    for images, labels in trainloader:
        images_train.append(images.view(images.size(0), -1))  # Làm phẳng hình ảnh
        labels_train.append(labels)
    
    images_train = torch.cat(images_train, dim=0)
    labels_train = torch.cat(labels_train, dim=0)
    
    images_train = images_train.numpy()
    labels_train = np.eye(10)[labels_train.numpy()]  # Chuyển đổi nhãn thành dạng one-hot

    # Cài đặt mô hình và tối ưu hóa
    optimizer = SGD(0.01)
    epochs = 20
    loss_func = CrossEntropy()

    softmax_regression = SoftmaxRegression(feature_dim=images_train.shape[1], num_classes=labels_train.shape[1], optimizer=optimizer, loss_func=loss_func)

    trainer = Trainer(softmax_regression, batch_size=64, epochs=epochs)
    trainer.train(images_train, labels_train)

    # Dự đoán trên tập kiểm tra
    images_test = []
    labels_test = []
    for images, labels in testloader:
        images_test.append(images.view(images.size(0), -1))  # Làm phẳng hình ảnh
        labels_test.append(labels)
    
    images_test = torch.cat(images_test, dim=0)
    labels_test = torch.cat(labels_test, dim=0)
    
    images_test = images_test.numpy()
    labels_test = np.eye(10)[labels_test.numpy()]  # Chuyển đổi nhãn thành dạng one-hot

    # Dự đoán trên tập kiểm tra
    pred = softmax_regression.predict(images_test)

    # In kết quả
    accuracy = np.mean(pred == labels_test.argmax(axis=1))
    print("Accuracy:", accuracy)

    print("Confusion matrix: ")
    print(confusion_matrix(labels_test.argmax(axis=1), pred))
