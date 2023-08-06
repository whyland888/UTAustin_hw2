from models import CNNClassifier, save_model
from utils import accuracy, load_data
import torch
import torch.optim as optim
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    # Hyperparameters
    batch_size = 128
    learning_rate = args.lr
    num_epochs = args.n_epochs

    # Data loading
    train_loader = load_data(r"/home/bojangles/Desktop/UT_Austin_NLP/UTAustin_hw2/data/train")
    valid_loader = load_data(r"/home/bojangles/Desktop/UT_Austin_NLP/UTAustin_hw2/data/valid")

    model = CNNClassifier()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training Loop
    epoch_accuracy = [0]
    for epoch in range(num_epochs):
        running_train_accuracy = []
        running_valid_accuracy = []
        train_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            # print(outputs)
            # print(outputs.shape)
            loss = criterion(outputs, labels)
            train_logger.add_scalar('loss', loss)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        overall_train_accuracy = torch.tensor(running_train_accuracy).mean().item()
        train_logger.add_scalar('accuracy', overall_train_accuracy)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in valid_loader:
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()

        overall_valid_accuracy = torch.tensor(running_valid_accuracy).mean().item()
        valid_logger.add_scalar('accuracy', overall_valid_accuracy)
        print(f"Validation Loss: {val_loss / len(valid_loader):.4f}")

        # Save if better than previous model
        # if overall_valid_accuracy > epoch_accuracy[-1]:
        #     save_model(model)
        # epoch_accuracy.append(overall_valid_accuracy)

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--n_epochs', type=int, required=True)

    args = parser.parse_args()
    train(args)
