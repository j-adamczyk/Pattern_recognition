from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from statistics import mean
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms as tfms, models

from dataset import AnimalsDataset


def train_val_split(dataset, val_size=0.3):
    # prepare constants, shuffle dataset
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_size * dataset_size))

    np.random.seed(0)
    np.random.shuffle(indices)

    # split dataset
    train_indices, val_indices = indices[split:], indices[:split]

    # count classes for training and validation to make sure they're balanced
    train_classes = dataset.iloc[train_indices, 1].to_numpy().flatten()
    val_classes = dataset.iloc[val_indices, 1].to_numpy().flatten()

    xs = [0, 1, 2]
    ys = np.bincount(train_classes)
    plt.bar(xs, ys)
    plt.title("Class counts for training")
    plt.savefig("class_count_train.png")
    plt.clf()

    ys = np.bincount(val_classes)
    plt.bar(xs, ys)
    plt.title("Class counts for validation")
    plt.savefig("class_count_val.png")
    plt.clf()

    train_set = dataset.iloc[train_indices, :].reset_index()
    val_set = dataset.iloc[val_indices, :].reset_index()

    return train_set, val_set


def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs=10, is_inception=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    for epoch in range(1, num_epochs + 1):
        print("Epoch", epoch)

        # train phase
        model.train()

        batch_losses = []
        batch_accuracies = []
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            if is_inception:
                outputs, aux_outputs = model(X)
                loss1 = criterion(outputs, y)
                loss2 = criterion(aux_outputs, y)
                loss = loss1 + 0.4 * loss2
            else:
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

        losses["train"].append(mean(batch_losses))
        accuracies["train"].append(mean(batch_accuracies))

        print("\tBatch train accuracy:", mean(batch_accuracies))

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

            print("\tBatch val accuracy:", mean(batch_accuracies))

        # if new validation accuracy is better, replace the best model
        if accuracies["val"][-1] > best_accuracy:
            best_accuracy = accuracies["val"][-1]
            best_model_weights = deepcopy(model.state_dict())

        if epoch in {10, 50}:
            xs = list(range(1, epoch + 1))
            plt.plot(xs, losses["train"], label="train")
            plt.plot(xs, losses["val"], label="val")
            plt.title(f"Losses ({epoch} epochs)")
            plt.legend(loc="upper right")
            plt.savefig(f"{epoch}_epochs_losses.png")
            plt.clf()

            plt.plot(xs, accuracies["train"], label="train")
            plt.plot(xs, accuracies["val"], label="val")
            plt.title(f"Accuracies ({epoch} epochs)")
            plt.legend(loc="lower right")
            plt.savefig(f"{epoch}_epochs_accuracies.png")
            plt.clf()
            print("After", epoch, "epochs, best accuracy:",
                  round(best_accuracy, 3))

    model.load_state_dict(best_model_weights)
    model.eval()
    return model


def try_mobilenet_v2():
    # values required by MobileNetV2
    transforms = tfms.Compose([
        tfms.Resize((224, 244)),
        tfms.ToTensor(),
        tfms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
    ])

    dataset = pd.read_csv("dataset.csv", delimiter=" ")
    train_csv, val_csv = train_val_split(dataset, val_size=0.3)

    # prepare dataset, samplers and loaders
    train_dataset = AnimalsDataset(train_csv, transforms=transforms)
    val_dataset = AnimalsDataset(val_csv, transforms=transforms)

    batch_size = 8
    num_workers = 4
    generator = torch.Generator().manual_seed(0)

    train_indices = list(range(len(train_dataset)))
    val_indices = list(range(len(val_dataset)))
    train_sampler = SubsetRandomSampler(train_indices, generator=generator)
    val_sampler = SubsetRandomSampler(val_indices, generator=generator)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers)

    # load model, freeze it for transfer learning, change last layer to 3
    # classes unfreezing it, send to GPU
    model = models.mobilenet_v2(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features,
                                     out_features=3,
                                     bias=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0)

    model = train_model(model, train_loader, val_loader, criterion,
                        optimizer, num_epochs=50, is_inception=False)
    torch.save(model.state_dict(), "mobilenet_v2.pth")


def try_inception_v3():
    # values required by InceptionV3
    transforms = tfms.Compose([
        tfms.Resize((299, 299)),
        tfms.ToTensor(),
        tfms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
    ])

    dataset = pd.read_csv("dataset.csv", delimiter=" ")
    train_set, val_set = train_val_split(dataset, val_size=0.3)

    # prepare dataset, samplers and loaders
    train_dataset = AnimalsDataset(train_set, transforms=transforms)
    val_dataset = AnimalsDataset(val_set, transforms=transforms)

    batch_size = 8
    num_workers = 4
    generator = torch.Generator().manual_seed(0)

    train_indices = list(range(len(train_dataset)))
    val_indices = list(range(len(val_dataset)))
    train_sampler = SubsetRandomSampler(train_indices, generator=generator)
    val_sampler = SubsetRandomSampler(val_indices, generator=generator)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers)

    # load model, freeze it for transfer learning, change last layer to 3
    # classes unfreezing it, send to GPU
    model = models.inception_v3(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features,
                         out_features=3,
                         bias=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0)

    model = train_model(model, train_loader, val_loader, criterion,
                        optimizer, num_epochs=50, is_inception=True)
    torch.save(model.state_dict(), "inception_v3.pth")


def try_inception_v3_with_aug():
    # values required by InceptionV3
    transforms = tfms.Compose([
        tfms.Resize((299, 299)),
        tfms.RandomHorizontalFlip(),
        tfms.RandomRotation(20),
        tfms.ToTensor(),
        tfms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
    ])

    dataset = pd.read_csv("dataset.csv", delimiter=" ")
    train_set, val_set = train_val_split(dataset, val_size=0.3)

    # prepare dataset, samplers and loaders
    train_dataset = AnimalsDataset(train_set, transforms=transforms)
    val_dataset = AnimalsDataset(val_set, transforms=transforms)

    batch_size = 8
    num_workers = 4
    generator = torch.Generator().manual_seed(0)

    train_indices = list(range(len(train_dataset)))
    val_indices = list(range(len(val_dataset)))
    train_sampler = SubsetRandomSampler(train_indices, generator=generator)
    val_sampler = SubsetRandomSampler(val_indices, generator=generator)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers)

    # load model, freeze it for transfer learning, change last layer to 3
    # classes unfreezing it, send to GPU
    model = models.inception_v3(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features,
                         out_features=3,
                         bias=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0)

    model = train_model(model, train_loader, val_loader, criterion,
                        optimizer, num_epochs=50, is_inception=True)
    torch.save(model.state_dict(), "inception_v3_aug.pth")


if __name__ == '__main__':
    #try_mobilenet_v2()
    #try_inception_v3()
    try_inception_v3_with_aug()
