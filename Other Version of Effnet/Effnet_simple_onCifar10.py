import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from math import ceil

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the data and reshape for CNN input
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
])

# Define EfficientNet components
def ConvBlock(filters, kernel_size, strides, padding='same'):
    return models.Sequential([
        layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU()
    ])

def MBBlock(in_channels, out_channels, kernel_size, strides, ratio, reduction=2):
    hidden_dim = in_channels * ratio
    reduced_dim = max(1, int(in_channels / reduction))
    
    def block(inputs):
        x = inputs
        if in_channels != hidden_dim:
            x = ConvBlock(hidden_dim, 1, 1)(x)

        x = ConvBlock(hidden_dim, kernel_size, strides, padding='same')(x)
        x = SqueezeExcitation(hidden_dim, reduced_dim)(x)
        x = layers.Conv2D(out_channels, 1, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        
        if strides == 1 and in_channels == out_channels:
            x = layers.Add()([x, inputs])  # Residual connection
        
        return x

    return block

def SqueezeExcitation(in_channels, reduced_dim):
    def block(inputs):
        x = layers.GlobalAveragePooling2D()(inputs)
        x = layers.Dense(reduced_dim, activation='relu')(x)
        x = layers.Dense(in_channels, activation='sigmoid')(x)
        return layers.multiply([inputs, x])
    return block

def EfficientNet(model_name, output):
    basic_mb_params = [
        [1, 16, 1, 1, 3],
        [6, 24, 2, 2, 3],
        [6, 40, 2, 2, 5],
        [6, 80, 3, 2, 3],
        [6, 112, 3, 1, 5],
        [6, 192, 4, 2, 5],
        [6, 320, 1, 1, 3],
    ]
    alpha, beta = 1.2, 1.1
    scale_values = {
        "b0": (0, 32, 0.2),  # Keeping resolution at 32x32 for CIFAR-10
        "b1": (0.5, 32, 0.2),
        "b2": (1, 32, 0.3),
        "b3": (2, 32, 0.3),
        "b4": (3, 32, 0.4),
        "b5": (4, 32, 0.4),
        "b6": (5, 32, 0.5),
        "b7": (6, 32, 0.5),
    }
    phi, resolution, dropout = scale_values[model_name]
    depth_factor, width_factor = alpha * phi, beta * phi
    last_channels = int(1280 * width_factor)

    inputs = layers.Input(shape=(32, 32, 3))
    x = data_augmentation(inputs)
    x = ConvBlock(int(32 * width_factor), 3, 1)(x)  # Stride 1 for small images
    in_channels = int(32 * width_factor)

    for k, c_o, repeat, s, n in basic_mb_params:
        out_channels = 4 * ceil(int(c_o * width_factor) / 4)
        num_layers = ceil(repeat * depth_factor)

        for layer in range(num_layers):
            stride = s if layer == 0 else 1
            x = MBBlock(in_channels, out_channels, n, stride, k)(x)
            in_channels = out_channels

    x = ConvBlock(last_channels, 1, 1)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(output, activation='softmax')(x)

    return models.Model(inputs, outputs)

# Initialize the model for CIFAR-10
model_name = 'b1'
output_class = 10
cifar_effnet = EfficientNet(model_name, output_class)

# Compile the model
cifar_effnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early Stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train the model
history = cifar_effnet.fit(
    x_train, y_train,
    batch_size=64,
    epochs=50,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping]
)

# Evaluate the model
test_loss, test_acc = cifar_effnet.evaluate(x_test, y_test)
print(f'Test Accuracy: {test_acc:.4f}')
