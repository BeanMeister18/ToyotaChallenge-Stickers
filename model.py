import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models

# Define the model
model = models.Sequential([
    layers.Flatten(input_shape=(64, 64, 3)),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Loading data
data = []
labels = []
directories = ['holes', 'stickers']  # replace with your directories

for i, directory in enumerate(directories):
    files = os.listdir("training/" +directory)
    for file in files:
        img = Image.open("training/" +directory + '/' + file)
        img = img.resize((64, 64))
        img = np.array(img)
        data.append(img)
        labels.append(i)

# Convert lists to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Normalize the data
data = data / 255.0

# Shuffle the data
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

# Train the model
model.fit(data, labels, epochs=50)

# Save the model
model.save('simple_ai_model.h5')
