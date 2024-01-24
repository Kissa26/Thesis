import glob
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.optimizers import Adam,RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.src.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from CategoriiModel import CategoriiModel
from SingleOutputModel import SingleOutputModel

# sa vad daca deosebeste tshits de tops si shirt
# clasificare shoes


def delete_invalid_images_shoes():
    df = pd.read_csv('data/myntradataset_articleTypeShoes/styles.csv')
    valid_categories = ['Casual Shoes', 'Flip Flops', 'Sandals', 'Formal Shoes', 'Casual Shoes', 'Flats',
                        'Sports Shoes', 'Heels']
    folder_path = 'data/myntradataset_articleTypeShoes/images'

    for filename in os.listdir(folder_path):
        img_name = os.path.splitext(filename)[0]
        img_id = int(img_name)
        if img_id in df['id'].values:
            img_category = df.loc[df['id'] == img_id, 'articleType'].values[0]
            if img_category not in valid_categories:
                img_to_delete = os.path.join(folder_path, filename)
                os.remove(img_to_delete)
                print(f'Deleted image: {img_to_delete}')

    df_filtered = df[df['articleType'].isin(valid_categories)]
    df_filtered.to_csv('C:/Users/Free/Documents/Licenta/filtered_shoes_articleType.csv', index=False)


def shirt_delete_invalid_images():
    df = pd.read_csv('data/myntradataset _articleType/styles.csv')
    valid_categories = ['Tshirt', 'Shirt', 'Tops']
    folder_path = 'data/myntradataset _articleType/images - Copy'
    for filename in os.listdir(folder_path):
        img_name = os.path.splitext(filename)[0]
        img_id = int(img_name)
        if img_id in df['id'].values:
            img_category = df.loc[df['id'] == img_id, 'articleType'].values[0]
            if img_category not in valid_categories:
                img_to_delete = os.path.join(folder_path, filename)
                os.remove(img_to_delete)
                print(f'Deleted image: {img_to_delete}')

    df_filtered = df[df['articleType'].isin(valid_categories)]
    df_filtered.to_csv('C:/Users/Free/Documents/Licenta/filtered_articleTypeShirt.csv', index=False)


def delete_invalid_images():
    df = pd.read_csv('data/myntradataset _articleType/styles.csv')
    df['articleType'] = df['articleType'].replace('Tshirt', 'Shirt', regex=True)
    df['articleType'] = df['articleType'].replace('Tops', 'Shirt', regex=True)
    valid_categories = ['Shirts', 'Blazers', 'Jeans', 'Sweatshirts', 'Dresses', 'Shorts', 'Trousers', 'Skirts']
    folder_path = 'data/myntradataset _articleType/images'
    for filename in os.listdir(folder_path):
        img_name = os.path.splitext(filename)[0]
        img_id = int(img_name)
        if img_id in df['id'].values:
            img_category = df.loc[df['id'] == img_id, 'articleType'].values[0]
            if img_category not in valid_categories:
                img_to_delete = os.path.join(folder_path, filename)
                os.remove(img_to_delete)
                print(f'Deleted image: {img_to_delete}')

    df_filtered = df[df['articleType'].isin(valid_categories)]
    df_filtered.to_csv('C:/Users/Free/Documents/Licenta/filtered_articleType.csv', index=False)


def load_and_process_images(name, path):
    df = pd.read_csv(name)
    encoded_articleType = pd.get_dummies(df['articleType'], prefix='articleType')

    images = [cv2.resize(cv2.imread(file), (80, 60)) for file in
              glob.glob(path) if
              cv2.imread(file) is not None]
    images_array = np.array(images)
    return images_array, encoded_articleType



def train_and_evaluate(images, labels):
    img_width = 80
    img_height = 60

    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2,
                                                                            random_state=42)
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.25,
                                                                          random_state=42)
    # Convertirea etichetelor în format one-hot
    train_labels_one_hot = to_categorical(train_labels, num_classes=7)
    val_labels_one_hot = to_categorical(val_labels, num_classes=7)
    test_labels_one_hot = to_categorical(test_labels, num_classes=7)
    model = SingleOutputModel().assemble_full_model(img_width, img_height, 7)
    init_lr = 1e-4
    epochs = 10
    opt = RMSprop(learning_rate=init_lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.save(
        'C:/Users/Free/Documents/Licenta/models/master_image_classification_model.h5')  # nu uita sa modifici cand schibi py pantofi

    data_gen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    history = model.fit(data_gen.flow(train_images, train_labels_one_hot, batch_size=32),
                        epochs=epochs,
                        validation_data=(val_images, val_labels_one_hot),
                        verbose=1)

    plt.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Model Accuracy(artycletype-shoes ) - {epochs} epochs, optimizer: RMSProp')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.savefig('C:/Users/Free/Documents/Licenta/ss/graphics/style_acc.png')
    model = tf.keras.saving.load_model('C:/Users/Free/Documents/Licenta/models/master_image_classification_model.h5')

    prediction = model.predict(test_images)
    true_classes = np.argmax(test_labels_one_hot, axis=1)
    predicted_classes = np.argmax(prediction, axis=1)
    accuracy = accuracy_score(true_classes, predicted_classes)
    print(f"Accuracy on test dataset: {accuracy * 100:.2f}%")


if __name__ == '__main__':
    name1 = 'C:/Users/Free/Documents/Licenta/filtered_shoes_articleType.csv'
    path1 = 'data/myntradataset_articleTypeShoes/images/*.jpg'
    name2 = 'C:/Users/Free/Documents/Licenta/filtered_articleType.csv'
    path2 = 'data/myntradataset _articleType/images/*.jpg'
    name3 = 'C:/Users/Free/Documents/Licenta/filtered_articleTypeShirt.csv'
    path3 = 'data/myntradataset _articleType/images - Copy/*.jpg'
    delete_invalid_images_shoes()
    delete_invalid_images()
    shirt_delete_invalid_images()
    images, labels = load_and_process_images(name1, path1)
    print(images.shape)
    print(labels.shape)
    train_and_evaluate(images, labels)
