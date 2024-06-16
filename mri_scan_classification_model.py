import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
import os

# Dataset: https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri?select=Testing

train_dir = 'MRI_Scans/Training'
test_dir = 'MRI_Scans/Testing'

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.5)

# Load and preprocess training data
train_data = train_datagen.flow_from_directory(
    train_dir,  # Directory containing the training data
    target_size=(224, 224),  # Resize all images to 224x224
    batch_size=32,  # Number of images to process at a time
    class_mode='categorical',  # Multiple classes, so use 'categorical'
    shuffle=True  # Shuffle the training data
)

# Load and preprocess the testing data (50% of data in the testing directory)
test_data = test_val_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False,
    subset='training'
)

# Load and preprocess the validation data (50% of data in the testing directory)
validation_data = test_val_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False,
    subset='validation'
)

# Visualize our training data
# Function takes in a parameter corresponding to the designated index of the class name

# Our categories are designated as:

# Glioma tumor: 0
# Meningioma tumor: 1
# No tumor: 2
# Pituitary tumor: 3

class_directory = {0 : 'glioma',
                   1 : 'meningioma',
                   2 : 'notumor',
                   3 : 'pituitary'}


def plot_random_scan(class_index):
  # Retrieve the folder path of the desired class
  directory_name = class_directory.get(class_index)

  folder_path = 'MRI_Scans/Training/' + directory_name

  # Iterates through files of the select directory and chooses a random scan
  files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
  random_file_path = os.path.join(folder_path, random.choice(files))

  # Displays the MRI scan
  mri_scan = mpimg.imread(random_file_path)
  plt.imshow(mri_scan)
  plt.title(directory_name)

plot_random_scan(random.randint(0,3))

model = tf.keras.Sequential([
    # First convolutional layer for input
    tf.keras.layers.Conv2D(filters=32,
                           kernel_size=(3, 3),
                           padding='same',
                           activation='relu',
                           input_shape=(224, 224, 3)),
    # Max pooling layer after convolution
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3),
                                 padding='valid'),


    # Convolution 2


    tf.keras.layers.Conv2D(filters=64,
                           kernel_size=(3, 3),
                           padding='same',
                           activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3),
                                 padding='valid'),


    # Convolution 3

    tf.keras.layers.Conv2D(filters=128,
                           kernel_size=(3, 3),
                           padding='same',
                           activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                 strides=2,
                                 padding='valid'),

    # Convolution 4

    tf.keras.layers.Conv2D(filters=256,
                           kernel_size=(2, 2),
                           padding='same',
                           activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                 strides=2,
                                 padding='valid'),

    # Convolution 5

    tf.keras.layers.Conv2D(filters=512,
                           kernel_size=(2, 2),
                           padding='same',
                           activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                 strides=2,
                                 padding='valid'),

    # Convolution 6

    tf.keras.layers.Conv2D(filters=1024,
                           kernel_size=(2, 2),
                           padding='same',
                           activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                 strides=2,
                                 padding='valid'),

    # After convolution, flatten the data since dense layers only support tensor shapes in 1 dimension
    tf.keras.layers.Flatten(),

    # Process output of convolution through fully connected layer for classification
    # Softmax activation since this is a categorical classification problem
    tf.keras.layers.Dense(4, activation='softmax')
])

# Compile the model
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(0.003),
              metrics=['categorical_accuracy'])

# Fit the model
history = model.fit(train_data,
                    epochs=20,
                    steps_per_epoch=len(train_data),
                    validation_data=validation_data,
                    validation_steps=len(validation_data))

# Create loss curves plot to visualize the performance of the model

def plot_loss_curves(history):
  train_loss = history.history['loss']
  val_loss = history.history['val_loss']

  train_accuracy = history.history['categorical_accuracy']
  val_accuracy = history.history['val_categorical_accuracy']

  epochs = range(len(history.history['loss']))

  plt.plot(epochs, train_loss, label='Training Loss')
  plt.plot(epochs, val_loss, label='Validation Loss')
  plt.title('Loss History')
  plt.xlabel('Epochs')
  plt.legend()

  plt.figure()
  plt.plot(epochs, train_accuracy, label='Training Accuracy')
  plt.plot(epochs, val_accuracy, label='Validation Accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend();

plot_loss_curves(history)

# Plots model predictions

def plot_model_predictions(model):
  image, label = test_data.next()

  index = random.randint(0, 31)

  model_prediction = model.predict(tf.expand_dims(image[index], axis=0))
  classification = class_directory.get(round(np.argmax(model_prediction)))

  plt.imshow(image[index])
  plt.title("Model Prediction: " + classification + "\nCorrect Class: " + class_directory.get(np.argmax(label[index])))

plot_model_predictions(model)
