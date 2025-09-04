# model.py
import os
import sys
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Use the correct model file that was saved during training
model = load_model('./model_digit_ocr_5.h5')

# Class labels
class_labels = ['0','1','2','3','4','5','6','7','8','9']


def classify_bubble(image):
    """Classify a single bubble image using the trained model"""
    try:
        # Ensure image is grayscale
        if len(image.shape) == 3:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            img = image.copy()
        
        # Resize to match training data
        img = cv2.resize(img, (64, 64))
        
        # Normalize the same way as training (rescale=1./255)
        img_array = img.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions to match model input shape (None, 64, 64, 1)
        img_array = img_array.reshape(1, 64, 64, 1)
        
        prediction = model.predict(img_array, verbose=0)
        label = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        return label, confidence
    except Exception as e:
        print(f"Classification error: {e}")
        return "Error", 0.0

def classify_batch(images):
    """
    Classify a list of bubble images using the trained model.
    Returns a list of (label, confidence) for each image.
    """
    try:
        preprocessed = []
        for img in images:
            # Ensure image is grayscale
            if len(img.shape) == 3:
                processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                processed = img.copy()
            
            # Resize to match training data
            resized = cv2.resize(processed, (64, 64))
            
            # Normalize the same way as training (rescale=1./255)
            arr = resized.astype(np.float32) / 255.0
            
            # Add channel dimension for grayscale (64, 64, 1)
            arr = arr.reshape(64, 64, 1)
            preprocessed.append(arr)

        batch_input = np.array(preprocessed)
        predictions = model.predict(batch_input, verbose=0)

        results = []
        for pred in predictions:
            label = class_labels[np.argmax(pred)]
            confidence = np.max(pred) * 100
            results.append((label, confidence))
        return results

    except Exception as e:
        print(f"Batch classification error: {e}")
        return [("Error", 0.0)] * len(images)
