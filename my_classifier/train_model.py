import tensorflow as tf
import lz4
import os
import cv2
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from numpy import asarray
from PIL import Image
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = "data"
CLASS_NAMES = ['Chris', 'Cong', 'Harry', 'RJ', 'Shawn']
    
def extract_face_from_image(image_path, required_size=(128, 128)):

  image = pyplot.imread(image_path)
  detector = MTCNN()
  faces = detector.detect_faces(image)

  face_images = []

  for face in faces:

    x1, y1, width, height = face['box']
    x2, y2 = x1 + width, y1 + height


    face_boundary = image[y1:y2, x1:x2]


    face_image = Image.fromarray(face_boundary)
    face_image = face_image.resize(required_size)
    face_array = asarray(face_image)
    face_images.append(face_array)

  return face_images

def load_data(data_dir):
    image_data = []
    labels = []
    class_names = os.listdir(data_dir)

    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)


            pixels = pyplot.imread(img_path)
            detector = MTCNN()
            faces = detector.detect_faces(pixels)
            extracted_face = extract_face_from_image(img_path)

            img_array = tf.keras.utils.img_to_array(cv2.resize(extracted_face[0],(128,128)))
            image_data.append(img_array)
            labels.append(idx)

    image_data = tf.convert_to_tensor(image_data) / 255.0

    labels = tf.convert_to_tensor(labels)
    CLASS_NAMES = class_names
    return image_data, labels, class_names

def create_model(class_names):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(class_names), activation='softmax')
    ])
    return model

def train_model_script():
    print("Loading data...")
    image_data, labels, class_names = load_data(DATA_DIR)
    print(f"Classes: {class_names}")

    image_data_np = image_data.numpy()
    labels_np = labels.numpy()

    X_train, X_val, y_train, y_val = train_test_split(image_data_np, labels_np, test_size=0.2, random_state=42)

    model = create_model(class_names)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("Training the model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )

    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {val_accuracy:.2f}")

    model.save("model.h5")
    print("Model saved as model.h5")