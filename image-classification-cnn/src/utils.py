import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def plot_training_history(history):
    """
    Plot training & validation accuracy/loss
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('../training_history.png')
    plt.show()

def get_class_labels(data_dir):
    """
    Automatically get class labels from data/train subdirectories
    """
    train_dir = os.path.join(data_dir, 'train')
    class_labels = sorted([name for name in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, name))])
    return class_labels

# Usage example in train.py: after training, call plot_training_history(history)
# In predict.py: class_labels = get_class_labels('../data')
