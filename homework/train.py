from models import CNNClassifier, save_model
from utils import accuracy, load_data
import torch
import torch_directml
import torch.optim as optim
import torch.utils.tensorboard as tb
import argparse


def train(args):
    from os import path
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    # Hyperparameters
    batch_size = args.batch_size
    learning_rate = args.lr
    num_epochs = args.n_epochs
    layers = args.layers

    # Paths to data
    local_train_path = r"C:\Users\Will\OneDrive\Desktop\State Farm\UT Austin Deep Learning\UTAustin_hw2\data\train"
    local_valid_path = r"C:\Users\Will\OneDrive\Desktop\State Farm\UT Austin Deep Learning\UTAustin_hw2\data\valid"
    colab_train_path = r"/content/UTAustin_hw2/data/train"
    colab_valid_path = r"/content/UTAustin_hw2/data/valid"

    # Data loading
    train_loader = load_data(local_train_path, batch_size=batch_size)
    valid_loader = load_data(local_valid_path, batch_size=batch_size)

    model = CNNClassifier(layers=layers).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=.90)
    elif args.optim == 'adamw':
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
        total_correct = 0.0
        total_samples = 0.0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()

                # Get accuracy
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()


        valid_logger.add_scalar('valid', val_loss/len(valid_loader), epoch)
        print(f"Validation Loss: {val_loss / len(valid_loader):.4f}")
        print(f"Accuracy: {total_correct/total_samples}")

        # Save if better than previous models
        if val_loss/len(valid_loader) < sorted(epoch_loss)[0]:
            save_model(model)
        epoch_loss.append(val_loss/len(valid_loader))



if __name__ == '__main__':

    device = torch_directml.device()  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--n_epochs', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument(
        '--optim',
        choices=['sgd', 'adamw'],
        help='Choose one of the available options: sgd, adamw'
    )
    parser.add_argument(
        '--activation',
        choices=['relu', 'leaky_relu'],
        help='Choose one of the available options: relu, leaky_relu'
    )
    parser.add_argument(
        '--layers',
        nargs='+',
        type=int,
        help='A list of integer values'
    )
    args = parser.parse_args()
    train(args)
