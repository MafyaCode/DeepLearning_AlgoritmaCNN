import os  # Untuk navigasi direktori/file
import cv2  # OpenCV untuk membaca dan memproses gambar
import numpy as np  # Untuk manipulasi array numerik (gambar)
# Mungkin perlu pandas jika membaca label dari CSV
# import pandas as pd
# Library spesifik framework DL Anda:
# from tensorflow.keras.preprocessing.image import ImageDataGenerator # Contoh Keras
# import torch # Contoh PyTorch
# from torchvision import datasets, transforms # Contoh PyTorch
# from torch.utils.data import DataLoader, Dataset # Contoh PyTorch

# Konsep Keras ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255, # Normalisasi
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    '../data/FER2013/train', # Path ke folder training
    target_size=(48, 48),     # Ukuran target gambar
    color_mode='grayscale',   # atau 'rgb'
    batch_size=32,            # Ukuran batch
    class_mode='categorical' # Untuk klasifikasi multi-kelas
)
# Lakukan hal serupa untuk validation data (biasanya tanpa augmentasi)

# Konsep PyTorch ImageFolder & DataLoader
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Definisikan transformasi (preprocessing & augmentasi)
train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # atau hapus jika RGB
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(), # Konversi ke Tensor PyTorch & normalisasi 0-1
    # transforms.Normalize(mean=[...], std=[...]) # Normalisasi lebih lanjut jika perlu
])

train_dataset = datasets.ImageFolder(
    '../data/FER2013/train', # Path ke folder training
    transform=train_transforms
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32, 
    shuffle=True # Acak data training
)
# Lakukan hal serupa untuk validation data (biasanya tanpa augmentasi acak)