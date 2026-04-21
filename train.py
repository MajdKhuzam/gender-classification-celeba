from tensorflow.keras import layers, models
from data_loader import load_splits
import os

DATA_DIR = os.path.join(__file__, 'Data', 'gender-recognition-200k-images-celeba', 'Dataset')

# Define the CNN model
model = models.Sequential([
    # Block 1
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Block 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Block 3
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    # Dense block
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),           # added dropout
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Load Data splits

train_images, train_labels, test_images, test_labels, val_images, val_labels = load_splits(DATA_DIR)

# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_data=(val_images, val_labels))