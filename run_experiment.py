from utils.model import MNIST_Net
from utils.train_utils import train_model, evaluate_model, get_mnist_dataset, DEVICE
from tqdm import tqdm


def run_exp(train_loader, test_loader, train_noise=0.4, n_epochs=5):
    model = MNIST_Net().to(DEVICE)
    model.set_noise_level(train_noise)

    # Train the model
    for _ in tqdm(range(n_epochs)):
        train_model(model, train_loader, test_loader, epochs=1, lr=0.0001, weight_decay=0)
    for _ in tqdm(range(n_epochs)):
        train_model(model, train_loader, test_loader, epochs=1, lr=0.00001, weight_decay=0)

    return model


if __name__ == '__main__':
    # Prepare the data loader
    train_loader, test_loader = get_mnist_dataset()

    # Run the baseline experiment
    print("Running regular/baseline training:")
    model = run_exp(train_loader, test_loader, train_noise=0)
    model.set_noise_level(0.4)
    eval_acc = evaluate_model(model, test_loader)
    print("Baseline evaluation acc", eval_acc)

    # Run the noise-aware experiment
    print("Running noise-aware training:")
    model = run_exp(train_loader, test_loader, train_noise=0.4)
    model.set_noise_level(0.4)
    eval_acc = evaluate_model(model, test_loader)
    print("Noisy-aware trainig evaluation acc", eval_acc)
