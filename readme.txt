CITATIONS
https://peerj.com/articles/cs-360/#p-13
https://www.kaggle.com/datasets/rodneypino/baybayin-and-latin-binary-images-in-mat-format
https://www.kaggle.com/datasets/jamesnogra/baybayn-baybayin-handwritten-images

FINDINGS:

(a) Convolutional Neural Network
    epoch: 16
    accuracy: 0.9816 - loss: 0.0677 - val_accuracy: 0.9782 - val_loss: 0.0775

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(50, 50, 1)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(17, activation='softmax')
    ])

(b) AlexNET
    epoch: 32
    accuracy: 0.9671 - loss: 0.1168 - val_accuracy: 0.9728 - val_loss: 0.0820 - learning_rate: 6.5610e-11
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(96, (11,11), strides=4, padding='same', activation='relu', input_shape=(227,227,1)),
        tf.keras.layers.MaxPooling2D((3,3), strides=(2,2), padding='valid'),

        tf.keras.layers.Conv2D(256, (3,3), strides=1, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((3,3), strides=(2,2), padding='valid'),

        tf.keras.layers.Conv2D(384, (3,3), strides=1, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(384, (3,3), strides=1, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(256, (3,3), strides=1, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((3,3), strides=(2,2), padding='valid'),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # dropout after dense
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # dropout after dense
        tf.keras.layers.Dense(17, activation='softmax')
    ])
