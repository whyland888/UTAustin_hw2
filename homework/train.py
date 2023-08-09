from models import CNNClassifier, save_model
from utils import accuracy, load_data
import torch
import torch.optim as optim
import torch.utils.tensorboard as tb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


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

    # Paths to data
    local_train_path = r"/home/bojangles/Desktop/UT_Austin_NLP/UTAustin_hw2/data/train"
    local_valid_path = r"/home/bojangles/Desktop/UT_Austin_NLP/UTAustin_hw2/data/valid"
    colab_train_path = r"UTAustin_hw2/data/train"
    colab_valid_path = r"UTAustin_hw2/data/valid"

    # Data loading
    train_loader = load_data(colab_train_path)
    valid_loader = load_data(colab_valid_path)

    model = CNNClassifier().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=.95)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Training Loop
    global_steps = 0
    epoch_loss = [100]
    for epoch in range(num_epochs):
        train_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_logger.add_scalar('train', loss, global_steps)
            loss.backward()
            optimizer.step()
            global_steps += 1

            train_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()

        valid_logger.add_scalar('valid', val_loss/len(valid_loader), global_steps)
        print(f"Validation Loss: {val_loss / len(valid_loader):.4f}")

        # Save if better than previous models
        if val_loss/len(valid_loader) < sorted(epoch_loss)[0]:
            save_model(model)
        epoch_loss.append(val_loss/len(valid_loader))



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--n_epochs', type=int, required=True)

    args = parser.parse_args()
    train(args)
