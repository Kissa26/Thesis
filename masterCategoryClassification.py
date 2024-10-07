import glob
import os
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

from BinaryModel import BinaryOutputModel
from keras.optimizers import Adam,RMSprop
from binaryModelPretrain import BinaryModel

def modify_csv():
    df = pd.read_csv('data/myntradataset_original/styles.csv')
    df['masterCategory'] = df['masterCategory'].apply(lambda x: x if x == 'Footwear' else 'Others')
    df.to_csv('data/myntradataset_masterCategory/masterCategory_filtrat.csv', index=False)
    count_footwear = df['masterCategory'].value_counts()['Footwear']
    print("Numărul de apariții pentru categoria 'Footwear':", count_footwear)


def delete_invalid_images():
    df = pd.read_csv('data/myntradataset_masterCategory/styles.csv')
    valid_categories = ['Apparel', 'Accessories']
    folder_path = 'data/myntradataset_masterCategory/images'
    for filename in os.listdir(folder_path):
        img_name = os.path.splitext(filename)[0]
        img_id = int(img_name)
        if img_id in df['id'].values:
            img_category = df.loc[df['id'] == img_id, 'masterCategory'].values[0]
            if img_category not in valid_categories:
                img_to_delete = os.path.join(folder_path, filename)
                os.remove(img_to_delete)
                print(f'Deleted image: {img_to_delete}')

    df_filtered = df[df['masterCategory'].isin(['Apparel', 'Accessories'])]
    df_filtered.to_csv('masterCategory_filtrat.csv', index=False)


def load_and_process_images_shoes():
    df = pd.read_csv('shoes_masterCategory.csv')
    df['binary_label'] = df['masterCategory'].apply(lambda x: 1 if x == 'Footwear' else 0)
    images = [cv2.resize(cv2.imread(file), (80, 60))
              for file in glob.glob("data/myntradataset_original/images/*.jpg")
              if cv2.imread(file) is not None]
    images_array = np.array(images)
    return images_array, df['binary_label']


def load_and_process_images():
    df = pd.read_csv('masterCategory_filtrat.csv')
    df['binary_label'] = df['masterCategory'].apply(lambda x: 1 if x == 'Apparel' else 0)
    images = [cv2.resize(cv2.imread(file), (80, 60))
              for file in glob.glob("data/myntradataset_masterCategory/images/*.jpg")
              if cv2.imread(file) is not None]

    images_array = np.array(images) / 255.0
    return images_array, df['binary_label']


def train_and_evaluate(images, labels,epochs):
    img_width = 60
    img_height = 80

    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.1,
                                                                            random_state=42)

    # Obține modelul final
    model = BinaryOutputModel.create_apparel_accessories_model()
    model1=BinaryModel.create_apparel_accessories_model()

    init_lr = 1e-2
    opt = RMSprop(learning_rate=init_lr)
    model1.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    model1.save('C:/Users/Free/Documents/Licenta/models/master1_image_classification_model.h5')

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    model.save('C:/Users/Free/Documents/Licenta/models/master_image_classification_model.h5')

    data_gen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.5, 1.5],
        fill_mode='nearest'
    )
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=20)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
    data_gen.fit(train_images)
    # Fit the data generator

    # Normalize input data
    scaler = StandardScaler()
    train_images_scaled = scaler.fit_transform(train_images.reshape(train_images.shape[0], -1)).reshape(
        train_images.shape)
    val_images_scaled = scaler.transform(val_images.reshape(val_images.shape[0], -1)).reshape(val_images.shape)

    # Train the model
    history = model.fit(
        data_gen.flow(train_images_scaled, train_labels, batch_size=64),
        epochs=epochs,
        validation_data=(val_images_scaled, val_labels),
        verbose=1,
        callbacks=[early_stopping, reduce_lr]
    )

    plt.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Model Accuracy (MasterCategory-shoes)- {epochs} epochs, optimizer:RMSprop')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == '__main__':

     #delete_invalid_images()
     #modify_csv()
     images, labels = load_and_process_images_shoes()
     #images, labels = load_and_process_images()
     #print(labels.shape)

     train_and_evaluate(images, labels,10)
