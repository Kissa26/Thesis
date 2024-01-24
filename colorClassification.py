import glob
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Dropout
from keras.layers import Flatten, Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def create_dataframe():
    df = pd.read_csv('data/myntradataset_color/styles.csv')
    categorii_valide = ['Blue', 'Silver', 'Black', 'Grey', 'Green',
                        'Purple', 'White', 'Beige', 'Brown', 'Pink',
                        'Red', 'Orange', 'Yellow', 'Gold']
    folder_path = 'data/myntradataset_color/images'
    # Parcurgeți fișierele din director
    for filename in os.listdir(folder_path):
        # Obțineți numele imaginii (fără extensia de fișier)
        nume_imag = os.path.splitext(filename)[0]
        nume_imag_cast = int(nume_imag)
        # Verificați dacă numele imaginii se află în coloana 'id' a fișierului CSV
        if nume_imag_cast in df['id'].values:
            # Obțineți categoria corespunzătoare din DataFrame
            categoria_imag = df.loc[df['id'] == nume_imag_cast, 'baseColour'].values[0]
            # Verificați dacă categoria nu este în lista de categorii valide
            if categoria_imag not in categorii_valide:
                # Construiți calea completă către imaginea de șters
                imagine_de_sters = os.path.join(folder_path, filename)
                # Ștergeți imaginea
                os.remove(imagine_de_sters)
                print(f'Imagine ștearsă: {imagine_de_sters}')
    # Aplicați codificarea one-hot pentru atributele 'color', 'material' și 'season'
    # Filtrarea rândurilor cu categoriile "Apparel" sau "Accessories"
    df_filtered = df[df['baseColour'].isin(categorii_valide)]
    # Salvarea datelor filtrate într-un nou fișier CSV
    df_filtered.to_csv('baseColour_filtrat.csv', index=False)

    encoded_color = pd.get_dummies(df_filtered['baseColour'], prefix='baseColour')

    # Combin datele codificate într-un nou DataFrame

    images = [cv2.resize(cv2.imread(file), (60, 80))
              for file in glob.glob("data/myntradataset_color/images/*.jpg")
              if cv2.imread(file) is not None]

    # Transform lista de imagini într-un numpy array
    images_array = np.array(images)

    return images_array, encoded_color


datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


class CuloriModel:
    def build_model(self, img_width, img_height, num_categories):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())

        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())

        model.add(Flatten())

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))

        model.add(Dense(num_categories, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model


if __name__ == '__main__':
    images, labels = create_dataframe()
    print(labels.shape, images.shape)

    img_width = 80
    img_height = 60

    # Split the data into train, validation, and test sets
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2,
                                                                            random_state=42)
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.25,
                                                                          random_state=42)

    # Build and compile the model
    culori_model = CuloriModel()
    model = culori_model.build_model(img_width, img_height, 14)
    init_lr = 1e-4
    epochs = 30
    opt = Adam(learning_rate=init_lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.save('image_classification_model.h5')
    # Train the model with data augmentation
    history = model.fit(datagen.flow(train_images, train_labels, batch_size=32),
                        epochs=epochs,
                        validation_data=(val_images, val_labels),
                        verbose=1
                        )

    # Plot training accuracy
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    model = tf.keras.saving.load_model('image_classification_model.h5')

    # Predict the class of the image
    prediction = model.predict(test_images)
    true_classes = np.argmax(test_labels, axis=1)

    # Convert one-hot encoded predictions to class labels
    predicted_classes = np.argmax(prediction, axis=1)

    # Calculate accuracy between true classes and predicted classes
    accuracy = accuracy_score(true_classes, predicted_classes)
    print(f"Acuratețea pe setul de date de test este: {accuracy * 100:.2f}%")
