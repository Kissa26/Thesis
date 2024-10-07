from keras.initializers import glorot_uniform
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense, Input
from keras.models import Model
from keras.regularizers import l2


class SingleOutputModel:
    def make_default_hidden_layers(self, inputs):
        x = Conv2D(32, (3, 3), activation="relu", padding="valid", kernel_initializer=glorot_uniform(seed=42))(inputs)
        x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        x = Conv2D(16, (3, 3), activation="relu", padding="valid", kernel_regularizer=l2(0.001))(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        x = Flatten()(x)
        x = Dense(64, activation="relu", kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(32, activation="relu", kernel_regularizer=l2(0.001))(x)
        return x

    def assemble_full_model(self, width, height, num_categories):
        input_shape = (height, width, 3)
        inputs = Input(shape=(height, width, 3))
        # Build hidden layers
        hidden_layers = self.make_default_hidden_layers(inputs)
        # Additional Dense layers

        x = Dense(64, activation="relu", kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(32, activation="relu", kernel_regularizer=l2(0.01))(x)
        category_output = Dense(num_categories, activation='softmax', name='category')(x)
        # Build the model
        model = Model(inputs=inputs, outputs=category_output, name="fashion_category_net")
        return model