import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the CIFAR-10 Dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Preprocess the Data
train_images, test_images = train_images / 255.0, test_images / 255.0

# Build the Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# Compile the Model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the Model
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Save the trained model
model.save("my_model.h5")
print("Model saved successfully.")

# Class Labels for CIFAR-10
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Function to classify a new image
def classify_new_image(image_path, model, class_labels):
    # Load the Image
    img = image.load_img(image_path, target_size=(32, 32))
    sample_image = image.img_to_array(img)
    sample_image = np.expand_dims(sample_image, axis=0)  # Add batch dimension
    sample_image /= 255.0  # Normalize the image

    # Make Predictions on the Sample Image
    predictions = model.predict(sample_image)

    # Get the predicted class
    predicted_class = np.argmax(predictions)

    # Print the predicted class label
    print(f"The image belongs to the class: {class_labels[predicted_class]}")

    # Display the Sample Image
    plt.imshow(img)
    plt.title(f"Predicted Label: {class_labels[predicted_class]}")
    plt.show()

# Test the Model with a New Image
new_image_path = "C:\\Users\\HP\\OneDrive\\Desktop\\Project\\images\\horse.jpg"
 
classify_new_image(new_image_path, model, class_labels)
