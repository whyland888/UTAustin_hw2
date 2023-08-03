from os import path
import torch
import torch.utils.tensorboard as tb
import numpy as np


def test_logging(train_logger, valid_logger):

    """
    Your code here.
    Finish logging the dummy loss and accuracy
    Log the loss every iteration, the accuracy only after each epoch
    Make sure to set global_step correctly, for epoch=0, iteration=0: global_step=0
    Call the loss 'loss', and accuracy 'accuracy' (no slash or other namespace)
    """

    # This is a strongly simplified training loop
    global_steps = 0
    for epoch in range(10):
        torch.manual_seed(epoch)
        #print(epoch)
        running_train_accuracy = []
        running_valid_accuracy = []
        for iteration in range(20):
            dummy_train_loss = 0.9**(epoch+iteration/20.)
            dummy_train_accuracy = epoch/10. + torch.randn(10).mean().item()
            running_train_accuracy.append(dummy_train_accuracy)
            train_logger.add_scalar('loss', dummy_train_loss, 20*epoch + iteration)
            global_steps += 1
        overall_train_accuracy = torch.tensor(running_train_accuracy).mean().item()
        print(overall_train_accuracy)
        train_logger.add_scalar('accuracy', overall_train_accuracy, global_steps)
        print(global_steps)
        global_steps += 1
        
        torch.manual_seed(epoch)
        for iteration in range(10):
            dummy_validation_accuracy = (epoch / 10. + torch.randn(10)).mean().item()
            running_valid_accuracy.append(dummy_validation_accuracy)
            #global_steps += 1
        overall_valid_accuracy = torch.tensor(running_valid_accuracy).mean().item()
        valid_logger.add_scalar('accuracy', overall_valid_accuracy, global_steps)
            


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('log_dir')
    args = parser.parse_args()
    train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
    valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'test'))
    test_logging(train_logger, valid_logger)
