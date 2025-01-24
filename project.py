import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import datasets, layers, models # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.layers import Input, Flatten, Dense # type: ignore
from tensorflow.keras.layers import Dropout # type: ignore
from tensorflow.keras.models import load_model # type: ignore

# Define the directory and categories
datadir = r"I:\Cs Project\Dataset"
categories = ['Glina', 'Lajthiza', 'Spring']
img_size = 128

# Function to load images
def load_images(datadir, categories, img_size):
    images = []
    labels = []
    for category in categories:
        path = os.path.join(datadir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                img_array = cv2.imread(img_path)
                if img_array is not None:
                    new_array = cv2.resize(img_array, (img_size, img_size))
                    images.append(new_array)
                    labels.append(class_num)
            except Exception as e:
                pass
    return np.array(images), np.array(labels)


images, labels = load_images(datadir, categories, img_size)


images = (images / 255.0).astype(np.float32)

#
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

class_names = ['Glina', 'Spring', 'Lajthiza']


# 1. Dense Layers Only
dropout_rate = 0.05  
dense_model = models.Sequential([
    Input(shape=(img_size, img_size, 3)),
    Flatten(),
    layers.Dense(512, activation='relu'),
    Dropout(dropout_rate), 
    layers.Dense(256, activation='relu'),
    Dropout(dropout_rate), 
    layers.Dense(len(categories), activation='softmax')
])
dense_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


dense_early_stopping = EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True)

dense_history = dense_model.fit(train_images, train_labels, epochs=25, batch_size=32, validation_data=(test_images, test_labels), callbacks=[dense_early_stopping])

dense_test_loss, dense_test_acc = dense_model.evaluate(test_images, test_labels, verbose=2)
print(f'Dense model test accuracy: {dense_test_acc}')
dense_model.save("dense_model.keras")


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(dense_history.history['accuracy'], label='Training Accuracy')
plt.plot(dense_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Dense Model: Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(dense_history.history['loss'], label='Training Loss')
plt.plot(dense_history.history['val_loss'], label='Validation Loss')
plt.title('Dense Model : Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# 2. Dense and Convolutional Layers
conv_dense_model = models.Sequential([
    Input(shape=(img_size, img_size, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(categories), activation='softmax')
])
conv_dense_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

conv_dense_early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)


conv_dense_history = conv_dense_model.fit(train_images, train_labels, epochs=20, batch_size=8, validation_data=(test_images, test_labels), callbacks=[conv_dense_early_stopping])

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(conv_dense_history.history['accuracy'], label='Training Accuracy')
plt.plot(conv_dense_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Conv-Dense Model: Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(conv_dense_history.history['loss'], label='Training Loss')
plt.plot(conv_dense_history.history['val_loss'], label='Validation Loss')
plt.title('Conv-Dense Model: Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

conv_dense_test_loss, conv_dense_test_acc = conv_dense_model.evaluate(test_images, test_labels, verbose=2)
print(f'Conv-Dense model test accuracy: {conv_dense_test_acc}')
conv_dense_model.save("conv_dense_model.keras")

# 3. Using a Pretrained Network
base_model = tf.keras.applications.MobileNetV2(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet')
base_model.trainable = False

pretrained_model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(categories), activation='softmax')
])
pretrained_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

pretrained_history = pretrained_model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

dense_model = load_model('dense_model.keras')
conv_dense_model = load_model('conv_dense_model.keras')
pretrained_model = load_model('pretrained_model.keras')

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(pretrained_history.history['accuracy'], label='Training Accuracy')
plt.plot(pretrained_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Pretrained Model: Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(pretrained_history.history['loss'], label='Training Loss')
plt.plot(pretrained_history.history['val_loss'], label='Validation Loss')
plt.title('Pretrained Model: Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

pretrained_test_loss, pretrained_test_acc = pretrained_model.evaluate(test_images, test_labels, verbose=2)
print(f'Pretrained model test accuracy: {pretrained_test_acc}')
pretrained_model.save("pretrained_model.keras")

# Load the pre-trained models
dense_model = load_model('dense_model.keras')
conv_dense_model = load_model('conv_dense_model.keras')
pretrained_model = load_model('pretrained_model.keras')


def load_and_preprocess_images(img_dir, img_size):
    images = []
    img_paths = []
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            img = img.astype(np.float32) / 255.0  
            images.append(img)
            img_paths.append(img_path)
    return np.array(images), img_paths


img_dir = "C:/Users/User/Desktop/bidona/Test"


images, img_paths = load_and_preprocess_images(img_dir, img_size)


plt.figure(figsize=(12, 12))
for i in range(len(images)):
    img = images[i]
    img_path = img_paths[i]

    prediction_dense = dense_model.predict(np.array([img]))
    prediction_conv_dense = conv_dense_model.predict(np.array([img]))
    prediction_pretrained = pretrained_model.predict(np.array([img]))

    index_dense = np.argmax(prediction_dense)
    index_conv_dense = np.argmax(prediction_conv_dense)
    index_pretrained = np.argmax(prediction_pretrained)

    plt.subplot(len(images), 3, i * 3 + 1)
    plt.imshow(img, cmap=plt.cm.binary)
    plt.title(f'Dense: {class_names[index_dense]}')
    plt.axis('off')

    plt.subplot(len(images), 3, i * 3 + 2)
    plt.imshow(img, cmap=plt.cm.binary)
    plt.title(f'Conv-Dense: {class_names[index_conv_dense]}')
    plt.axis('off')

    plt.subplot(len(images), 3, i * 3 + 3)
    plt.imshow(img, cmap=plt.cm.binary)
    plt.title(f'Pretrained: {class_names[index_pretrained]}')
    plt.axis('off')

plt.tight_layout()
plt.show()
