import os
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('simple_ai_model.h5')

# Function to preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((64, 64))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    return img / 255.0

# Paths to testing directories
testing_directories = {
  'holes': 'testing/holes',
  'stickers': 'testing/stickers'
}

# Testing
total = 0
correct = 0

for class_name, directory in testing_directories.items():
    files = os.listdir(directory)
    for file in files:
        image_path = os.path.join(directory, file)
        processed_image = preprocess_image(image_path)
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions)
        if (class_name == 'holes' and predicted_class == 0) or (class_name == 'stickers' and predicted_class == 1):
            correct += 1
        total += 1

# Calculate accuracy
accuracy = correct / total * 100
print(f'Accuracy: {accuracy:.2f}%')
