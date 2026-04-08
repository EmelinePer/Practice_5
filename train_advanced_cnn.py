import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_advanced_cnn():
    print("Loading MNIST Dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize to 0-1 and reshape
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

    print("Building Advanced CNN Model...")
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Classification Head
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("Setting up Data Augmentation...")
    # This is CRITICAL for recognizing handwritten numbers drawn by mice
    # It adds rotations, zooming, and shifting during training
    datagen = ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.15,
        height_shift_range=0.15
    )
    datagen.fit(x_train)

    print("Starting Training (Will take a few minutes)...")
    # Train the model with augmented data
    model.fit(
        datagen.flow(x_train, y_train, batch_size=64),
        epochs=15, 
        validation_data=(x_test, y_test)
    )

    print("Saving Advanced Model...")
    model.save('mnist_cnn_advanced.h5')
    print("Saved as mnist_cnn_advanced.h5!")

if __name__ == "__main__":
    train_advanced_cnn()
