from keras.initializers import glorot_uniform
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense, Input
from keras.models import Model
from keras.regularizers import l2


class BinaryOutputModel:
    def make_default_hidden_layers(self, inputs):
        x = Conv2D(32, (3, 3), activation="relu", padding="valid", kernel_initializer=glorot_uniform(seed=42))(inputs)
        x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        x = Conv2D(16, (3, 3), activation="relu", padding="valid")(x)#, kernel_regularizer=l2(0.01))(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        x = Flatten()(x)
        x = Dense(64, activation="relu")(x)#, kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)#0,5
        x = Dense(32, activation="relu")(x)#, kernel_regularizer=l2(0.01))(x)
        return x

    def assemble_full_model(self, width, height):
        input_shape = (height, width, 3)
        inputs = Input(shape=input_shape)

        # Build hidden layers
        hidden_layers = self.make_default_hidden_layers(inputs)

        # Add additional Dense layers
        x = Flatten()(hidden_layers)
        x = Dense(64, activation="relu")(x)#, kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)#0,5
        x = Dense(32, activation="relu")(x)#, kernel_regularizer=l2(0.01))(x)

        # Output layer - 1 neuron pentru clasificarea între 'Apparel' și 'Accessories'
        output = Dense(1, activation='sigmoid', name='binary_output')(x)

        # Build the model
        model = Model(inputs=inputs, outputs=output, name="fashion_net")
        return model
