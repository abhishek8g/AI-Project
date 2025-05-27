# 1. Project Content
The project is a preliminary step in a Cat vs. Dog image classification task. It involves mounting a Google Drive, accessing a dataset of cat images, and displaying a few sample images to verify the data loading process.

# 2. Project Code
The Python code from the Jupyter Notebook is as follows:

Cell 1: Mount Google Drive

Python

from google.colab import drive
drive.mount('/content/drive')
Cell 2: Define Cat Dataset Path

Python

cat='/content/drive/MyDrive/Dataset/cat'
Cell 3: Load and Display Cat Images

Python

import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image


folder_path = '/content/drive/MyDrive/Dataset/cat'


image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]


for i in range(min(5, len(image_files))):
    img_path = os.path.join(folder_path, image_files[i])
    img = image.load_img(img_path, target_size=(128, 128))

    plt.imshow(img)
    plt.title(f"Image: {image_files[i]}")
    plt.axis('off')
    plt.show()

base_dir = "/content/drive/MyDrive/Dataset"
cat_dir = os.path.join(base_dir, "cat")
dog_dir = os.path.join(base_dir, "dog")

img_size = (150, 150)
batch_size = 32


cat_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, shear_range=0.2, horizontal_flip=True)
dog_datagen = ImageDataGenerator(rescale=1./255)

cat_data = cat_datagen.flow_from_directory(
    cat_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

dog_data = dog_datagen.flow_from_directory(
    dog_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


base_dir = '/content/drive/MyDrive/Dataset'
train_dir = base_dir
validation_dir = base_dir # For simplicity, using same directory for train/validation. In a real scenario, split your dataset.


train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2) # 80% for training, 20% for validation

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128), # Consistent with your image loading in the original notebook
    batch_size=32,
    class_mode='binary', # 'binary' for 2 classes (cat/dog)
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    validation_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid') # Sigmoid for binary classification
])


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


history = model.fit(
    train_generator,
    epochs=10, # You can adjust the number of epochs
    validation_data=validation_generator
)

model.summary()
import numpy as np
from tensorflow.keras.preprocessing import image
from lime import lime_image
import matplotlib.pyplot as plt


img_path = '/content/drive/MyDrive/Dataset/cat/1.jpg' # Example image path - REPLACE WITH A REAL IMAGE FILE
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
img_array = img_array / 255.0 # Normalize as done during training


def predict_fn(images):
    # Ensure images are normalized to 0-1 range
    # LIME passes normalized images (0-1 range) if hide_color=0, but it's good practice to ensure
    normalized_images = images # LIME with hide_color=0 already normalizes to 0-1
    predictions = model.predict(normalized_images)
    # For binary classification, LIME expects two columns (e.g., [prob_class_0, prob_class_1])
    # Assuming 0 is 'cat' and 1 is 'dog' based on class_mode='binary' and flow_from_directory default ordering
    return np.hstack([1 - predictions, predictions])


explainer = lime_image.LimeImageExplainer()


explanation = explainer.explain_instance(
    img_array[0].astype('double'), # LIME expects a single image (no batch dimension) and double type
    predict_fn,
    top_labels=1,
    hide_color=0,
    num_samples=1000
)


predicted_class_index = explanation.top_labels[0]

temp, mask = explanation.get_image_and_mask(
    predicted_class_index,
    positive_only=False, # Show both positive and negative contributions
    num_features=5, # Number of superpixels to highlight
    hide_rest=True
)


plt.imshow(img_array[0])
plt.title("Original Image")
plt.axis('off')
plt.show()


plt.imshow(temp)
plt.imshow(mask, cmap='viridis', alpha=0.5)
plt.title(f"LIME Explanation for Class: {predicted_class_index} ({'dog' if predicted_class_index == 1 else 'cat'})") # Add class name to title
plt.axis('off')
plt.show()




# 3. Key Technologies

The key technologies used in this project are:

Google Colab: For running the Jupyter Notebook and mounting Google Drive.
Python: The programming language used for the project.
TensorFlow/Keras: For image preprocessing (tensorflow.keras.preprocessing.image), indicating that a deep learning model will likely be built using this framework.
Matplotlib: For plotting and displaying the images.
OS Module: For interacting with the file system to list the image files.
# 4. Description
The project begins by mounting a Google Drive to access the dataset. It then defines the path to a folder containing images of cats. The main part of the code imports the necessary libraries for file handling, image plotting, and image preprocessing. It then lists all the image files (with .jpg, .jpeg, or .png extensions) in the specified folder and displays the first five images with their filenames as titles. This is a common initial step in a deep learning project to visually inspect the data before building a model.

# 5. Output
The notebook produces the following outputs:

A message indicating that the Google Drive has been successfully mounted.
A series of plots, each displaying one of the first five cat images from the dataset folder.


# 6. Further Research
Based on the initial code, the following steps would be logical for further research and development of the project:

Load and Preprocess the Dog Dataset: The same process for loading and displaying cat images should be applied to the dog dataset.
Create Labels: Create labels for the images (e.g., 0 for cats, 1 for dogs) to be used for training the model.
Build a Convolutional Neural Network (CNN): Design and build a CNN model using TensorFlow/Keras to classify the images.
Train and Evaluate the Model: Train the CNN model on the cat and dog images and evaluate its performance using metrics like accuracy, precision, and recall.
Data Augmentation: To improve the model's accuracy and prevent overfitting, apply data augmentation techniques like rotation, flipping, and zooming to the training images.
Transfer Learning: Use a pre-trained model (like VGG16, ResNet, or MobileNet) and fine-tune it on the cat and dog dataset to potentially achieve better results with less training time.
Deployment: Deploy the trained model as a web application or an API to allow users to upload an image and get a prediction of whether it's a cat or a dog.
