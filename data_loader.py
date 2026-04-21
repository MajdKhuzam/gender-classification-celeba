import os
import cv2
import numpy as np

# Function to load images and labels from folders names
def load_data(data_dir, img_width, img_height):
    images = []
    labels = []
    label_mapping = {'female': 0, 'male': 1}
    for folder in os.listdir(data_dir):
        if folder.startswith('.'):  # Skip hidden folders
            continue
        folder_path = os.path.join(data_dir, folder)
        label = label_mapping[folder.lower()]  # Get the integer label
        for file in os.listdir(folder_path):
            if file.endswith('.jpg'):
                image_path = os.path.join(folder_path, file)
                image = cv2.imread(image_path)
                image = cv2.resize(image, (img_width, img_height))  # Resize images
                images.append(image)
                labels.append(label)
    return np.array(images), np.array(labels)

def load_splits(data_dir):
    # Load and preprocess the dataset
    train_path = os.path.join(data_dir, 'Train')
    train_images, train_labels = load_data(train_path,128,128)
    train_labels = train_labels.astype(np.int32)
    print("Train data loaded successfully")

    test_path = os.path.join(data_dir, 'Test')
    test_images, test_labels = load_data(test_path,128,128)
    test_labels = test_labels.astype(np.int32)
    print("Test data loaded successfully")

    val_path = os.path.join(data_dir, 'Validation')
    val_images, val_labels = load_data(val_path,128,128)
    val_labels = val_labels.astype(np.int32)
    print("Validation data loaded successfully")
    
    return train_images, train_labels, test_images, test_labels, val_images, val_labels