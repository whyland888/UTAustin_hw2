import torch
import torch.nn.functional as F


class CNNClassifier(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1, activation='relu'):
            super().__init__()
            if activation == 'relu':
                activation_layer = torch.nn.ReLU()
            elif activation == 'leaky_relu':
                activation_layer = torch.nn.LeakyReLU()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride),
                activation_layer,
                torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1),
                activation_layer
            )

        def forward(self, x):
            return self.net(x)

    def __init__(self, layers=[32,64,128], n_input_channels=3, n_classes=6, activation='relu'):
        super().__init__()
        if activation == 'relu':
            activation_layer = torch.nn.ReLU()
        elif activation == 'leaky_relu':
            activation_layer = torch.nn.LeakyReLU()
        L = [torch.nn.Conv2d(n_input_channels, 32, kernel_size=7, padding=3, stride=2),
            activation_layer,
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        c = 32
        for l in layers:
            L.append(self.Block(c, l, stride=2))
            c = l 
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(layers[-1], n_classes)

    def forward(self, x):
        z = self.network(x)  # compute features
        z = z.mean(dim=[2, 3])  # global average pooling
        return self.classifier(z)



def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, CNNClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
    raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = CNNClassifier()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th'), map_location='cpu'))
    return r
