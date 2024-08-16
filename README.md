
# Baybayin Handwritten Script Recognition using Tensorflow and CV2

Baybayin Handwritten Script Recognition using Tensorflow and CV2 is a machine learning project that aims to recognize and classify handwritten Baybayin characters using deep learning techniques. Baybayin is an ancient script used in the Philippines, and this project seeks to preserve and promote the script by developing a system that can accurately recognize and interpret handwritten Baybayin characters.

![](demo.gif)

## Run Locally

1.) Clone the project

```bash
  git clone https://github.com/clarencelubrin/baybayin-script-recognition
```
2.) Go to the directory

```bash
  cd baybayin-script-recognition
```

3.) Open app.py using python

```bash
  python app.py
```


## Findings
(a) Convolutional Neural Network
    epoch: 16
    accuracy: 0.9816 - loss: 0.0677 - val_accuracy: 0.9782 - val_loss: 0.0775

```Python
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(50, 50, 1)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(17, activation='softmax')
    ])
```
(b) AlexNET
    epoch: 32
    accuracy: 0.9671 - loss: 0.1168 - val_accuracy: 0.9728 - val_loss: 0.0820 - learning_rate: 6.5610e-11
```Python
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
```
## References

Pino, R., et. al. (2021). Optical character recognition system for Baybayin scripts using support vector machine. Retrieved from: https://peerj.com/articles/cs-360/#p-13

Mendoza, R., et. al. (2022). Block-level Optical Character Recognition System
for Automatic Transliterations of Baybayin Texts
Using Support Vector Machine. Retrieved from: https://philjournalsci.dost.gov.ph/images/pdf/pjs_pdf/vol151no1/block_level_optical_character_recognition_system_.pdf

Pino, R. (2021). Baybayin and Latin (Binary) Images in .mat Format. Retrieved from: https://www.kaggle.com/datasets/rodneypino/baybayin-and-latin-binary-images-in-mat-format

Nogra, J. (2019). Baybayín (Baybayin) Handwritten Image. Retrieved from: https://www.kaggle.com/datasets/jamesnogra/baybayn-baybayin-handwritten-images
