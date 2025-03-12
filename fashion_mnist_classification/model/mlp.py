import torch.nn as nn
import torch
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size = 784, hidden_size1 = 256, hidden_size2 = 128, output_size = 10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size1)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size2)
    
    def forward(self,x,returnfc3=False):
        x = self.fc1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x) 
        return x

if __name__ == '__main__':
    model = MLP()
    x = torch.randn(1, 784)
    print(model)
    print(model(x).shape)