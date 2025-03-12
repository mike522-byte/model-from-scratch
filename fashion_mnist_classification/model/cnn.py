import torch.nn as nn
import torch
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 3, stride = 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.fc1 = nn.Linear(64 * 13 * 13, 128)
        self.relu = nn.LeakyReLU(0.01)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
if __name__ == '__main__':
    model = CNN()
    x = torch.randn(32, 1, 28, 28)
    print(model)
    print(model(x).shape)
