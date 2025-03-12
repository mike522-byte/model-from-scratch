import torch
import torch.optim as optim
from model.cnn import CNN
import numpy as np
import argparse
from data import data_loader
import torch.nn.init as init

def init_weights(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        init.kaiming_uniform_(m.weight, nonlinearity='relu')  # He Initialization
        if m.bias is not None:
            init.zeros_(m.bias)  # Bias initialized to zero

def test(model, test_loader, val_loader, criterion, device, isTest = True):
    model.eval()
    data_loader = test_loader if isTest else val_loader
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
    
    test_accuracy = 100 * correct_test / total_test
    test_loss = test_loss / len(data_loader)
    if isTest:
        print(f"Total Loss: {test_loss:.4f}, Total Accuracy: {test_accuracy:.4f}%")
    else:
        print(f"Validation Loss: {test_loss:.4f}, Validation Accuracy: {test_accuracy:.4f}%")



def train(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    model.train()  # Set model to training mode
    for epoch in range(epochs):
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # forwardpropagation
            outputs = model(images) # Flatten input
            loss = criterion(outputs, labels)

            optimizer.zero_grad()  # Zero gradients
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        accuracy = 100 * correct_train / total_train
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {accuracy:.2f}%")

        if (epoch + 1)%3 == 0:
            test(model, _, val_loader, criterion, device, isTest=False)

    return model

parser = argparse.ArgumentParser(description='MLP')
parser.add_argument('--epochs', type=int, default=15, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
args = parser.parse_args()

learning_rate = args.lr
epochs = args.epochs
batch_size = args.batch_size

model = CNN()
# kaiming weight initialization for relu activation
model.apply(init_weights)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()
train_loader , val_loader, test_loader= data_loader(batch_size=batch_size)


model = train(model, train_loader, val_loader, criterion, optimizer, device, epochs)
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, "cnn_model.pth")

test(model, test_loader, val_loader, criterion, device, isTest=True)

