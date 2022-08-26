import torch.nn.functional as F
import torch.optim as optim
import torch
from torchvision import datasets, transforms
import numpy as np

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_mnist_dataset(digit_1=3, digit_2=5):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('data', train=True, download=True, transform=transform)
    dataset_test = datasets.MNIST('data', train=False, transform=transform)

    idx = (torch.tensor(dataset_train.targets) == digit_1) + (torch.tensor(dataset_train.targets) == digit_2)
    dataset_train.targets[dataset_train.targets == digit_1] = 0
    dataset_train.targets[dataset_train.targets == digit_2] = 1
    dataset_train = torch.utils.data.dataset.Subset(dataset_train, np.where(idx == 1)[0])

    idx = (torch.tensor(dataset_test.targets) == digit_1) + (torch.tensor(dataset_test.targets) == digit_2)
    dataset_test.targets[dataset_test.targets == digit_1] = 0
    dataset_test.targets[dataset_test.targets == digit_2] = 1
    dataset_test = torch.utils.data.dataset.Subset(dataset_test, np.where(idx == 1)[0])

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=256, num_workers=1, pin_memory=True,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=2048, num_workers=1, pin_memory=True,
                                              shuffle=True)
    return train_loader, test_loader


def train_model(model, train_loader, test_loader, epochs, lr=0.001, weight_decay=0.1, constraint_epoch=0,
                alpha=40):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.to(DEVICE)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = F.huber_loss(output.view(-1, ), target.view(-1, ).float()) + model.get_weight_loss(alpha)
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item()
            if epoch > constraint_epoch:
                model.apply_constraints()
        eval_acc = evaluate_model(model, test_loader)
        print("Epoch: ", epoch, "eval acc: ", eval_acc, "train loss: ", total_loss)

    return model


def evaluate_model(model, test_loader):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output > 0.5  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().detach().item()
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    return accuracy
