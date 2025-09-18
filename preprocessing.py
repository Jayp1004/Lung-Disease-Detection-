"""
Preprocessing utilities for lung disease detection project
"""

import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (150, 150)
BATCH_SIZE = 32

def create_data_generators(base_dir):
    """
    Creates train, val, and test generators from chest_xray dataset
    """
    train_path = os.path.join(base_dir, "train")
    val_path = os.path.join(base_dir, "val")
    test_path = os.path.join(base_dir, "test")

    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        zoom_range=0.2,
        shear_range=0.2,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(rescale=1.0/255)
    test_datagen = ImageDataGenerator(rescale=1.0/255)

    train_gen = train_datagen.flow_from_directory(
        train_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary"
    )

    val_gen = val_datagen.flow_from_directory(
        val_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary"
    )

    test_gen = test_datagen.flow_from_directory(
        test_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )

    return train_gen, val_gen, test_gen
