import tensorflow as tf
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
from .train_model import CLASS_NAMES


model = tf.keras.models.load_model('model.h5')   


def extract_face_from_image(image_path, required_size=(128, 128)):

    image = Image.open(image_path)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image = np.array(image)
    detector = MTCNN()
    faces = detector.detect_faces(image)

    face_images = []

    for face in faces:
        x1, y1, width, height = face['box']
        x2, y2 = x1 + width, y1 + height

        face_boundary = image[y1:y2, x1:x2]

        face_image = Image.fromarray(face_boundary)
        face_image = face_image.resize(required_size)
        face_array = np.asarray(face_image)
        face_images.append(face_array)

    return face_images

def predict_image(image_path):
    
    extracted_face = extract_face_from_image(image_path)
    if extracted_face:  
        img_array = tf.keras.utils.img_to_array(extracted_face[0])
        img_array = np.expand_dims(img_array, axis=0) / 255.0 

        if img_array.shape[-1] != 3:
            raise ValueError(f"Expected image with 3 channels, but got {img_array.shape[-1]} channels")

        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        predicted_class = CLASS_NAMES[predicted_class_index]

        return predicted_class
    else:
        return "No face detected"