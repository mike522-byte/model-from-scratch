import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("Eiffel-Tower-Paris-resized-600x398.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

pixels = img_rgb.reshape(-1, 3).astype(np.float32)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# Number of colors for quantization
num_colors = 10 

# Apply K-Means clustering
_, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
quantized_image = centers[labels.flatten()].reshape(img_rgb.shape)

# Display original and quantized images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(quantized_image)
plt.title(f"Quantized Image ({num_colors} Colors)")
plt.axis("off")

plt.show()

