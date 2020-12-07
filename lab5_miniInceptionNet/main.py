from copy import deepcopy
from statistics import mean
from time import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from mini_inception_net import MiniInceptionNet


def get_default_params():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = CIFAR10(root='./data',
                        train=True,
                        download=True,
                        transform=transform)
    train_loader = DataLoader(train_set,
                              batch_size=32,
                              shuffle=True)

    val_set = CIFAR10(root='./data',
                      train=False,
                      download=True,
                      transform=transform)
    val_loader = DataLoader(val_set,
                            batch_size=32,
                            shuffle=False)

    criterion = nn.CrossEntropyLoss()
    return train_loader, val_loader, criterion


def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs=10):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters:", num_params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    best_model_weights = deepcopy(model.state_dict())
    best_accuracy = 0

    losses = {
        "train": [],
        "val": []
    }

    accuracies = {
        "train": [],
        "val": []
    }

    training_times = []

    for epoch in range(1, num_epochs + 1):
        print("Epoch", epoch)

        # train phase
        model.train()

        batch_losses = []
        batch_accuracies = []
        start = time()
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            # check training accuracy; change to eval mode to freeze dropout
            with torch.no_grad():
                model.eval()
                outputs = model(X)
                y_pred = torch.max(outputs, dim=1)[1]
                loss = criterion(outputs, y)

                # change loss, labels and predictions to CPU types (float and
                # Numpy arrays)
                loss = float(loss.detach().cpu())
                y = y.cpu().numpy()
                y_pred = y_pred.detach().cpu().numpy()
                batch_losses.append(loss)
                batch_accuracies.append(accuracy_score(y, y_pred))
                model.train()

        end = time()
        losses["train"].append(mean(batch_losses))
        accuracies["train"].append(mean(batch_accuracies))
        training_times.append(end - start)

        # evaluation phase; change to eval mode to freeze dropout
        model.eval()
        with torch.no_grad():
            batch_losses = []
            batch_accuracies = []
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                outputs = model(X)
                _, y_pred = torch.max(outputs, dim=1)
                loss = criterion(outputs, y)

                # change loss, labels and predictions to CPU types (float and
                # Numpy arrays)
                loss = float(loss.detach().cpu())
                y = y.cpu().numpy()
                y_pred = y_pred.detach().cpu().numpy()
                batch_losses.append(loss)
                batch_accuracies.append(accuracy_score(y, y_pred))

            losses["val"].append(mean(batch_losses))
            accuracies["val"].append(mean(batch_accuracies))
            print("   acc:", mean(batch_accuracies))
            print("   epoch time:", training_times[-1])

        # if new validation accuracy is better, replace the best model
        if accuracies["val"][-1] > best_accuracy:
            best_accuracy = accuracies["val"][-1]
            best_model_weights = deepcopy(model.state_dict())

    xs = list(range(1, num_epochs + 1))
    plt.plot(xs, losses["train"], label="train")
    plt.plot(xs, losses["val"], label="val")
    plt.title(f"Losses")
    plt.legend(loc="upper right")
    plt.savefig(f"losses.png")
    plt.clf()

    plt.plot(xs, accuracies["train"], label="train")
    plt.plot(xs, accuracies["val"], label="val")
    plt.title(f"Accuracies")
    plt.legend(loc="lower right")
    plt.savefig(f"accuracies.png")
    plt.clf()
    print("Best accuracy:", round(best_accuracy, 3))

    print("Average epoch time:", round(mean(training_times), 3))

    model.load_state_dict(best_model_weights)
    model.eval()
    return model


def train():
    train_loader, test_loader, criterion = get_default_params()
    model = MiniInceptionNet(num_classes=10, use_separable=False)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=20)

    torch.save(model.state_dict(), "mini_inception_net.pth")


def try_different_input_size():
    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MiniInceptionNet(num_classes=10, use_separable=False)
    model.load_state_dict(torch.load("mini_inception_net.pth"))
    model = model.eval().to(device)

    # data
    transform = transforms.Compose([
        transforms.Resize((36, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    val_set = CIFAR10(root='./data',
                      train=False,
                      download=True,
                      transform=transform)

    img1, label1 = val_set[1]
    img2, label2 = val_set[2]
    img3, label3 = val_set[3]

    # reshape images to single-image 4D batches, concatenate them along the
    # 0-th (batch size) dimension to get 3 images batch
    images = torch.cat([img.to(device).unsqueeze(0)
                        for img in [img1, img2, img3]])
    labels = torch.Tensor([label1, label2, label3])

    outputs = torch.argmax(model(images), dim=1).cpu().numpy()

    val_set = CIFAR10(root='./data',
                      train=False,
                      download=True,
                      transform=transforms.transforms.Resize((36, 28)))

    img1 = val_set[1][0]
    img2 = val_set[2][0]
    img3 = val_set[3][0]

    img1.save("img1.png")
    img2.save("img2.png")
    img3.save("img3.png")

    print(label1, outputs[0])
    print(label2, outputs[1])
    print(label3, outputs[2])


def get_CAMs(upsample_size, feature_conv, weight, class_indices):
    _, nc, h, w = feature_conv.shape
    CAMs = []
    for i, class_index in enumerate(class_indices):
        cam = weight[class_index].dot(feature_conv[i].reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        CAMs.append(cv2.resize(cam_img, (upsample_size, upsample_size)))
    return CAMs


def plot_heatmap():
    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MiniInceptionNet(num_classes=10, use_separable=False)
    model.load_state_dict(torch.load("mini_inception_net.pth"))
    model = model.eval().to(device)

    features = []

    def hook_feature(module, input, output):
        features.append(output.detach().cpu().data.numpy())

    model._modules.get("fourth_inception_2").register_forward_hook(hook_feature)
    weights = model.fc_linear.weight.detach().cpu().numpy()

    # data
    transform = transforms.Compose([
        transforms.Resize((36, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    val_set = CIFAR10(root='./data', train=False, download=True)

    # make images bigger, they're too small by default (28x28)
    up_size = 140

    img1_original = cv2.resize(np.array(val_set[1][0]),
                               dsize=(up_size, up_size))
    img2_original = cv2.resize(np.array(val_set[2][0]),
                               dsize=(up_size, up_size))
    img3_original = cv2.resize(np.array(val_set[3][0]),
                               dsize=(up_size, up_size))

    val_set = CIFAR10(root='./data', train=False, download=True,
                      transform=transform)
    img1 = val_set[1][0]
    img2 = val_set[2][0]
    img3 = val_set[3][0]

    images = torch.cat([img.to(device).unsqueeze(0)
                        for img in [img1, img2, img3]])

    probas = model(images)
    features = features[0]
    idx = probas.argmax(dim=1).cpu().numpy()

    # plot Class Activation Heatmap (CAM) on each image
    CAMs = get_CAMs(up_size, features, weights, idx)

    heatmap1 = cv2.applyColorMap(CAMs[0], cv2.COLORMAP_JET)
    heatmap2 = cv2.applyColorMap(CAMs[1], cv2.COLORMAP_JET)
    heatmap3 = cv2.applyColorMap(CAMs[2], cv2.COLORMAP_JET)

    img1 = (heatmap1 * 0.3 + img1_original * 0.5).astype(np.uint8)
    img2 = (heatmap2 * 0.3 + img2_original * 0.5).astype(np.uint8)
    img3 = (heatmap3 * 0.3 + img3_original * 0.5).astype(np.uint8)

    cv2.imwrite("heatmap1.png", img1)
    cv2.imwrite("heatmap2.png", img2)
    cv2.imwrite("heatmap3.png", img3)


if __name__ == '__main__':
    # train()
    # try_different_input_size()
    plot_heatmap()
