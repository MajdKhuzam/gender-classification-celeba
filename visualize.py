import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import os
from data_loader import load_data
from evaluate import predicted_labels

# DATA_DIR = os.path.join(__file__, '../Data', 'gender-recognition-200k-images-celeba', 'Dataset')
DATA_DIR = '/home/majd/Desktop/Gender Classification CNN - CelebA/Data/gender-recognition-200k-images-celeba/Dataset'
TEST_PATH = os.path.join(DATA_DIR, 'Test')

test_images, test_labels = load_data(TEST_PATH,128,128)
test_labels = test_labels.astype(np.int32)

# Helper dictionary to map integers back to text labels
label_mapping_reverse = {0: 'Female', 1: 'Male'}

# Set up a 3x3 grid for plotting
plt.figure(figsize=(10, 10))

for i in range(9):
    # Pick a random image from the test set
    idx = random.randint(0, len(test_images) - 1)
    
    image = test_images[idx]
    true_label = label_mapping_reverse[test_labels[idx]]
    pred_label = label_mapping_reverse[predicted_labels[idx]]
    
    plt.subplot(3, 3, i + 1)
    
    # OpenCV loads in BGR, convert to RGB for Matplotlib visualization
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    
    # Color the title green if correct, red if incorrect
    title_color = 'green' if true_label == pred_label else 'red'
    plt.title(f"True: {true_label}\nPred: {pred_label}", color=title_color)
    plt.axis('off')

plt.tight_layout()
plt.show()