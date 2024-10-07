import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.optimizers import SGD, RMSprop, Adadelta
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.layers import GlobalAveragePooling2D, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from keras.applications import MobileNetV2, InceptionV3,VGG16
from keras.models import Model  # Asigură că importăm clasa Model din pachetul keras.models
from sklearn.preprocessing import StandardScaler

from  BinaryModel import BinaryOutputModel
from SingleOutputModel import SingleOutputModel

def create_pretrained_model(img_shape, num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=img_shape, alpha=0.35)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def create_dataframe():
    df = pd.read_csv('data/myntradataset_season/styles.csv')
    season_mapping = {'Summer': 'warm', 'Spring': 'warm', 'Winter': 'cold', 'Fall': 'cold'}

    # Înlocuirea valorilor din coloana 'season' folosind dicționarul definit anterior
    df['season'] = df['season'].replace(season_mapping)
    encoded_season = pd.get_dummies(df['season'], prefix='season')
    warm_percentage = (encoded_season['season_warm'].sum() / len(encoded_season)) * 100

    # Combin datele codificate într-un nou DataFrame
    images = [cv2.resize(cv2.imread(file), (80, 60))
              for file in glob.glob("data/myntradataset_season/images/*.jpg")
              if cv2.imread(file) is not None]

    # Transform lista de imagini într-un numpy array
    images_array = np.array(images)
    return images_array, encoded_season,warm_percentage



def train_and_evaluate(train_images, train_labels, val_images, val_labels, opt,epochs,name):
    img_width = 80
    img_height = 60
    num_classes = 2 # Numărul de clase

    # Redimensionarea imaginilor la dimensiunile așteptate de modelul preantrenat MobileNetV2
    train_images = np.array([cv2.resize(img, (img_width, img_height)) for img in train_images])
    val_images = np.array([cv2.resize(img, (img_width, img_height)) for img in val_images])



    # Crearea modelului preantrenat MobileNetV2 și congelarea stratelor
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model1=BinaryOutputModel.create_apparel_accessories_model()
    model1.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


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
        featurewise_std_normalization=True  # Normalizare a valorilor pixelilor
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
    history = model1.fit(
        train_images_scaled,
        train_labels,
        epochs=epochs,
        validation_data=(val_images_scaled, val_labels),
        verbose=1,
        callbacks=[early_stopping, reduce_lr]
    )

    # Plot training accuracy
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

    plt.title(f'Model Accuracy(seson) - {epochs} epochs, optimizer:{name}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    plt.savefig('C:/Users/Free/Documents/Licenta/ss/graphics/season_acc.png')


if __name__ == '__main__':
    images, labels,warm_percentage = create_dataframe()
    print("Procentajul de etichete 'warm' din setul de date este: {:.2f}%".format(warm_percentage))

    print(labels.shape)

    img_width = 80
    img_height = 60
    opt1 = Adam(learning_rate=1e-4)
    opt2 = RMSprop(learning_rate=1e-4)
    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.3,
                                                                            random_state=42)


    train_and_evaluate(train_images, train_labels, val_images, val_labels,opt1,10,"Adam")
    train_and_evaluate(train_images, train_labels, val_images, val_labels, opt1, 80,"Adam")

    train_and_evaluate(train_images, train_labels, val_images, val_labels,opt2,10,"RMSprop")
    train_and_evaluate(train_images, train_labels, val_images, val_labels, opt2, 80,"RMSprop")


