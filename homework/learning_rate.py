from models import CNNClassifier
from utils import load_data
import torch
import torch.optim as optim
import torch.utils.tensorboard as tb


def find_lr(args):
    from os import path
    lr_logger = tb.SummaryWriter(path.join(args.log_dir, 'learning_rate'))

    # Hyperparameters
    batch_size = 256
    learning_rates = torch.range(.001, .1, .005)
    n_epochs = 2

    # Data loading
    train_loader = load_data(r"/home/bojangles/Desktop/UT_Austin_NLP/UTAustin_hw2/data/train",
                             batch_size=batch_size)

    criterion = torch.nn.CrossEntropyLoss()

    for learning_rate in learning_rates:
        model = CNNClassifier()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=.95)
        loss_difference = []
        for epoch in range(n_epochs):
            train_loss = 0
            for i, (images, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                lr_logger.add_scalar('loss', loss)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            loss_difference.append(train_loss)
            lr_logger.add_scalar('accuracy', train_loss)
            print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {train_loss / len(train_loader):.4f}, Learning Rate: {learning_rate}")
        print(f"Difference in loss between epochs: {loss_difference[0] - loss_difference[1]}")




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')

    args = parser.parse_args()
    find_lr(args)