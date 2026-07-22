import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max_pool(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool(x)

        x = x.reshape(x.shape[0], -1)

        x = self.fc1(x)
        x = self.relu(x)
        out = self.fc2(x)
        return out