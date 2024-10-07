import glob
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam, RMSprop, SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from CategoriiModel import CategoriiModel
from ArticleTypeShoesCustomCNNModel import CustomCNNModel
from SingleOutputModel import SingleOutputModel

from keras.losses import SparseCategoricalCrossentropy


# Shoes Flip Flops Sandal
# nu uuita sa faci pt pantofi


def create_dataframe(categorii_valide, path):
    df = pd.read_csv(path + '/styles.csv')
    folder_path = path + '/images'
    # Parcurgeți fișierele din director
    for filename in os.listdir(folder_path):
        # Obțineți numele imaginii (fără extensia de fișier)
        nume_imag = os.path.splitext(filename)[0]
        nume_imag_cast = int(nume_imag)
        # Verificați dacă numele imaginii se află în coloana 'id' a fișierului CSV
        if nume_imag_cast in df['id'].values:
            # Obțineți categoria corespunzătoare din DataFrame
            categoria_imag = df.loc[df['id'] == nume_imag_cast, 'subCategory'].values[0]
            # Verificați dacă categoria nu este în lista de categorii valide
            if categoria_imag not in categorii_valide:
                # Construiți calea completă către imaginea de șters
                imagine_de_sters = os.path.join(folder_path, filename)
                # Ștergeți imaginea
                os.remove(imagine_de_sters)
                print(f'Imagine ștearsă: {imagine_de_sters}')
    # Aplicați codificarea one-hot pentru atributele 'color', 'material' și 'season'
    # Filtrarea rândurilor cu categoriile "Apparel" sau "Accessories"
    df_filtered = df[df['subCategory'].isin(categorii_valide)]
    # Salvarea datelor filtrate într-un nou fișier CSV
    df_filtered.to_csv('subCategory_filtrat.csv', index=False)

    encoded_masterCategory = pd.get_dummies(df_filtered['subCategory'], prefix='subCategory')

    # Combin datele codificate într-un nou DataFrame

    images = [cv2.resize(cv2.imread(file), (80, 60))
              for file in glob.glob(path + "/images/*.jpg")
              if cv2.imread(file) is not None]

    # Transform lista de imagini într-un numpy array
    images_array = np.array(images)

    return images_array, encoded_masterCategory


def train_test(shoes, epochs, opt, name):
    df = pd.read_csv('data/myntradataset_original/styles.csv')

    numar_atribute_unice = df['usage'].nunique()
    print(numar_atribute_unice)

    valid1 = ['Shoes', 'Flip Flops', 'Sandal']
    valid2 = ['Topwear', 'Bottomwear', 'Dress''Topwear', 'Bottomwear', 'Dress']
    path1 = 'data/myntradataset _subCategory_shoes'
    path2 = 'data/myntradataset _subCategory'
    if shoes == True:
        images, labels = create_dataframe(valid1, path1)
    else:
        images, labels = create_dataframe(valid2, path2)
    print(labels.shape)

    img_width = 80
    img_height = 60

    # Numărul total de clase pentru atribute (culoare, material, sezon)

    # Split the data into train, validation, and test sets

    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2,
                                                                          random_state=42)

    # Build and compile the model
    input_shape = (60, 80, 3)
    x=SingleOutputModel()
    model=x.assemble_full_model(img_width,img_height,3)
    data_gen = ImageDataGenerator(
        rotation_range=40,  # Rotirea imaginii într-un interval de 40 de grade
        width_shift_range=0.25,  # Deplasarea imaginii pe orizontală cu maximum 20%
        height_shift_range=0.25,  # Deplasarea imaginii pe verticală cu maximum 20%
        shear_range=0.25,  # Deformații aleatorii ale imaginii cu maximum 20%
        zoom_range=0.3,  # Zoom aleatoriu în interiorul imaginii cu maximum 20%
        horizontal_flip=True,  # Răsturnare orizontală aleatoare a imaginii
        fill_mode='nearest',  # Modul de completare a pixelilor după transformări
        brightness_range=[0.8, 1.2],  # Modificarea luminozității între 80% și 120%
        channel_shift_range=20,  # Schimbarea aleatorie a valorilor canalelor de culoare
        featurewise_center=True,  # Tăiere aleatoare a imaginilor

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

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    # Train the model
    # Train the model
    history = model.fit(
        data_gen.flow(train_images_scaled, train_labels, batch_size=64),
        epochs=epochs,
        validation_data=(val_images_scaled, val_labels),
        verbose=1,
        callbacks=[early_stopping, reduce_lr]
    )
    # Plot training accuracy
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    if shoes == True:
        plt.title(f'Model Accuracy (subCategory shoes)- {epochs} epochs, optimizer:{name}')
    else:
        plt.title(f'Model Accuracy (subCategory)- {epochs} epochs, optimizer:{name}')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    init_lr = 0.0002

    opt1 = Adam(learning_rate=init_lr)
    opt2 = RMSprop(learning_rate=init_lr)

    #train_test(True, 10, opt1, "Adam")
    #train_test(True, 80, opt1, "Adam")
    #train_test(False, 10, opt1, "Adam")
    #train_test(False, 80, opt1, "Adam")

    train_test(True, 10, opt2, "RMSprop")
    train_test(True, 80, opt2, "RMSprop")
    train_test(False, 10, opt2, "RMSprop")
    train_test(False, 80, opt2, "RMSprop")
