#pip install tensorflow
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import tensorflow as tf
import keras
from keras import layers

img_width = 128 # 130 # ширина input'a
img_height = 128 # 50 # высота input'a
def encode_img(img, img_type = '*.png'):
    # Preprocessing
    #    We will use 3 channels input (in original code were 1 channel)
    if img_type == '*.jpeg':
        img = tf.io.decode_jpeg(img, channels=3, dct_method='INTEGER_ACCURATE')
    #        img = tf.image.rgb_to_grayscale(img)
    else:
        img = tf.io.decode_png(img, channels=3)

    # Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Resize to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    # Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    return img

def encode_single_sample(img_path, label):
    # Read image
    img = tf.io.read_file(img_path)
    # Preprocessing
    img = encode_img(img)
    # Map the characters in label to numbers
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    # Return a dict as our model is expecting two inputs
    return {"image": img, "label": label}
def split_data(images: 'list', labels: 'list', train_size=0.9, shuffle=True):
    size = len(images)
    # 1. Make an indices array and shuffle it, if required
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)
    # 2. Get the size of training samples
    train_samples = int(size * train_size)
    # 3. Split data into training and validation sets
    x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
    x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
    return x_train, x_valid, y_train, y_valid

x_train, x_valid, y_train, y_valid = split_data(np.array(images), np.array(labels))
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = (
    train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
)

validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
validation_dataset = (
    validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
)
# Loss function
class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred, **kwargs):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred
def build_model():
    # Inputs to the model
    input_img = layers.Input(
        # Lets use RGB input instead of black and white
        shape=(img_width, img_height, 3), name="image", dtype="float32"
    )
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    # First conv block
    x = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block
    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # We have used two max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing the output to the RNN part of the model
    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    # RNNs
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    # Output layer
    x = layers.Dense(
        len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2"
    )(x)

    # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model
    model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
    )
    # Optimizer
    opt = keras.optimizers.Adam()
    # Compile the model and return
    model.compile(optimizer=opt)
    return model
model = build_model()
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
)
print('R RUUUU UUU NN')
def train_run():
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=100,
        callbacks=[early_stopping],
    )
    return history
history = train_run()

def train_test():
    model: Functional = keras.models.load_model(MODEL_FNAME)
    images = list(map(str, [i for i in data_dir_test.glob(img_type)]))
    images = sorted(images, key=lambda *h: random.random())
    labels = [
        img.split(os.path.sep)[-1].split('-')[0].split('.jpeg')[0].ljust(max_length)[:max_length]
        for img in images
    ]

    test_dataset = tf.data.Dataset.from_tensor_slices((numpy.array(images), numpy.array(labels)))
    test_dataset = (
        test_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    good = 0
    total = 0

    for i in test_dataset.take(1000):
        label, image = i['label'], i['image']
        preds = model.call(image)
        pred_texts = decode_batch_predictions(preds)[0]
        orig_texts = [tf.strings.reduce_join(num_to_char(l)).numpy().decode("utf-8").replace('[UNK]', '') for l in
                      label]
        if need_visualization and total == 0:
            visualize()
        for i in range(len(pred_texts)):
            success = pred_texts[i] == orig_texts[i]
            if success:
                good += 1
            total += 1
        print(f"Progress {total / len(images):.2%}[{total}/{len(images)}]...", end='\r')
    print(f"Success: {(good / total):.2%}\n")