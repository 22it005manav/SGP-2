import numpy as np
import pickle
import cv2
import os

# Function to load CIFAR-10 data batches
def load_cifar10_batch(file):
    with open(file, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)  # Reformat the images
        return X, Y

# Path to the CIFAR-10 dataset batches
dataset_path = './dataset/cifar-10-batches-py'

# Load all training batches
train_images = []
for i in range(1, 6):
    batch_file = os.path.join(dataset_path, f'data_batch_{i}')
    X_batch, Y_batch = load_cifar10_batch(batch_file)
    train_images.append(X_batch)

# Combine all training images
train_images = np.concatenate(train_images, axis=0)

# Convert images to grayscale for colorization
grayscale_images = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in train_images])

# Save grayscale images to a folder
for idx, img in enumerate(grayscale_images[:10]):  # Save only 10 images for simplicity
    img_path = f'./images/img_{idx}.jpg'
    cv2.imwrite(img_path, img)
    print(f"Saved {img_path}")
