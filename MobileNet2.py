import tensorflow as tf
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Dropout




class MobileNetV2Model:
    def __init__(self, input_shape, num_classes, alpha=1.0):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.alpha = alpha
        self.model = self._build_model()

    def _build_model(self):
        # Load the pre-trained MobileNetV2 model without top layers
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=self.input_shape, alpha=self.alpha)

        # Freeze the convolutional layers of the pre-trained model
        for layer in base_model.layers:
            layer.trainable = False

        # Add new convolutional and dense layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)

        # Create the final model
        model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

        # Compile the model
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model
