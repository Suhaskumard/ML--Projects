import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

IMG_SIZE = 128

model = tf.keras.models.load_model('../models/cnn_model.h5')

# Load class labels - Update based on your dataset classes
class_labels = ['class1', 'class2']  # Replace with your actual class names, e.g., ['cat', 'dog']

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    return predicted_class, np.max(prediction)  # Return class and confidence

# Example usage
if __name__ == "__main__":
    result = predict_image('test.jpg')  # Replace with your test image path
    print(f"Predicted: {result[0]} (confidence: {result[1]:.2f})")
