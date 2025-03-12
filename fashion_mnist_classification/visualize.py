import torch
import matplotlib.pyplot as plt
from model.mlp import MLP
from model.cnn import CNN
from data import data_loader
import seaborn as sns
import numpy as np

model = MLP()
# Load the checkpoint dictionary
checkpoint = torch.load('mlp_model.pth', weights_only=True)  # Ensure safe loading
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Extract weights of the last layer (FC3) corresponding to class predictions
weights_fc3 = model.fc3.weight.data.cpu().numpy()
weights_fc2 = model.fc2.weight.data.cpu().numpy()
weights_fc1 = model.fc1.weight.data.cpu().numpy()

# fc1 learns the spatial pattern of the input image
fig1, axes = plt.subplots(5, 5, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    if i < 25:  # Show only 25 neurons
        img = weights_fc1[i].reshape(28, 28)  # Reshape to 28x28
        ax.imshow(img, cmap="magma")
        ax.axis("off")
plt.suptitle("Visualization of FC1 Weights (First 25 Neurons)", fontsize=14)

# fc3 maps neurons to output classes
plt.figure(figsize=(10, 5)) 
sns.heatmap(weights_fc3, cmap="viridis", center=0)
plt.title("Heatmap of FC3 Weights (10 Classes x 128 Neurons)")
plt.xlabel("FC2 Neurons")
plt.ylabel("Classes (0-9)")
plt.show()

# what if i feed an image through the network and visualize the product of the input and the weight
_ , _, test_loader= data_loader()
with torch.no_grad():
    images,labels = next(iter(test_loader))
    single_image = images[0].view(1,-1)
    image_show = single_image.reshape(1,28,28).squeeze()
    output = model(single_image)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image_show, cmap="gray")
axes[0].set_title("Input Image")
sns.heatmap(output.T, ax = axes[1], square=True)
axes[1].set_title("Output Logits")
plt.show()

model_cnn = CNN()
checkpoint_cnn = torch.load('cnn_model.pth', weights_only=True)  # Ensure safe loading
model_cnn.load_state_dict(checkpoint_cnn['model_state_dict'])
model_cnn.eval()

def visualize_conv_weights(layer, title, cols=8):
    weights = layer.weight.data.cpu().numpy()  # Extract weights
    num_filters = weights.shape[0]  # Number of filters
    rows = (num_filters // cols) + int(num_filters % cols > 0)  # Auto-calculate rows

    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    axes = axes.flatten()  # Flatten axes array for easy iteration

    for i in range(rows * cols):
        if i < num_filters:
            filter_img = weights[i, 0]  # Extract single-channel filter
            filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min())  # Normalize for better contrast
            axes[i].imshow(filter_img, cmap="gray")
        axes[i].axis("off")  # Remove axis labels

    plt.suptitle(title, fontsize=14)
    plt.show()

visualize_conv_weights(model_cnn.conv1, "Conv1 Filters")