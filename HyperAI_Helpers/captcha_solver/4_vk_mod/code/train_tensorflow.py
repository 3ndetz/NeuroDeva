
"""
## Introduction
This example demonstrates a simple OCR model built with the Functional API. Apart from
combining CNN and RNN, it also illustrates how you can instantiate a new layer
and use it as an "Endpoint layer" for implementing CTC loss. For a detailed
guide to layer subclassing, please check out
[this page](https://keras.io/guides/making_new_layers_and_models_via_subclassing/)
in the developer guides.
"""
# УСТАНОВКА !
# 1 скачиваем в докер

# docker pull tensorflow/tensorflow:latest-gpu-jupyter  # latest release w/ GPU support and Jupyter

# 2 запускаем докер контейнер

# docker run -v F:/Onix/Downloads/minebot/1Chatbot_TF/HyperAI_Helpers/captcha_solver/4_vk_mod/code:/tf/ --gpus all -p 8888:8888 -d -it --name TensorflowTrainer tensorflow/tensorflow:latest-gpu-jupyter

# 3 доп библиотеки (устанавливать в docker)

# pip install colorama

# для конверта onnx (СМ. ПРИМЕЧАНИЕ ВАЖНО! ПРИ УСТАНОВКЕ НАДО МОДИФИЦИРОВАТЬ СКАЧАННУЮ БИБЛЮ)
# pip install git+https://github.com/microsoft/onnxconverter-common
# pip install onnxruntime
# pip install pip install -U tf2onnx
#
#ПРИМЕЧАНИЕ ДЛЯ КОНВЕРТА onnx https://github.com/onnx/tensorflow-onnx/issues/2172
# нужно найти файл библиотеки tf2onnx python3.9/site-packages/tf2onnx/convert.py (ПРОЩЕ ИСКАТЬ ПО ВЫСКАЧИВШЕЙ ОШИБКЕ)
#   ЗАМЕНИТЬ graph_captures = concrete_func.graph._captures  # pylint: disable=protected-access
#   НА:
#    captured_inputs = [t_name.name for t_val, t_name in graph_captures.values()]
#    input_names = [input_tensor.name for input_tensor in concrete_func.inputs
#                   if input_tensor.name not in captured_inputs]
#    output_names = [output_tensor.name for output_tensor in concrete_func.outputs
#                    if output_tensor.dtype != tf.dtypes.resource]
# для теста на onnx
# pip install opencv-python-headless

#    python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

import os
import time
import os
import numpy as np
###import matplotlib.pyplot as plt

from pathlib import Path

#import tensorflow as tf
#from tensorflow import keras
from keras import layers
from utils.config import *

from tools import *

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
def split_data(train_images: 'np.ndarray', train_labels: 'np.ndarray',
               test_images: 'np.ndarray'=None, test_labels: 'np.ndarray'=None,
               train_size=0.9, shuffle=True): #train_size=0.9

    # 1. Get the total size of the dataset
    size = len(train_images)
    # 2. Make an indices array and shuffle it, if required
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)
    # 3. Get the size of training samples
    train_samples = int(size * train_size)
    # 4. Split data into training and validation sets
    x_train, y_train = train_images[indices[:train_samples]], train_labels[indices[:train_samples]]
    x_valid, y_valid = train_images[indices[train_samples:]], train_labels[indices[train_samples:]]
    print(f'[SPLIT DATASET LOAD] [TRAIN] loaded {str(size)} from TRAIN dir')
    if test_images is not None and test_labels is not None:
        np.hstack((x_valid, test_images))
        np.hstack((y_valid, test_labels))
        # EXTEND THE ARRAY
        #x_valid = (test_images)
        #y_valid.extend(test_labels)
        print(f'[SPLIT DATASET LOAD] [TEST] loaded {str(size - train_samples)} from TRAIN dir +{str(len(test_images))} from TEST dir; result={str(len(x_valid))}')
    return x_train, x_valid, y_train, y_valid
def build_model():
    # Inputs to the model
    input_img = layers.Input(
        # let's use RGB input
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


# Get list of all the images
images = sorted(list(map(str, [i for i in data_dir.glob(img_type)])))
labels = [
    img.split(os.path.sep)[-1].split('_')[0].split('.')[0].ljust(max_length)[:max_length]
    #img.split(os.path.sep)[-1].split('-')[0].split('.')[0].ljust(8)[:8]
    for img in images
]

images_for_test = sorted(list(map(str, [i for i in data_dir_test.glob(img_type)])))
labels_for_test = [
    img.split(os.path.sep)[-1].split('_')[0].split('.')[0].ljust(max_length)[:max_length]
    #img.split(os.path.sep)[-1].split('-')[0].split('.')[0].ljust(8)[:8]
    for img in images_for_test
]


print(f"Number of TRAIN images found:{colorama.Fore.CYAN}", len(images), colorama.Fore.RESET)
print(f"Number of TRAIN labels found:{colorama.Fore.CYAN}", len(labels), colorama.Fore.RESET)
print(f"Number of TEST images found:{colorama.Fore.CYAN}", len(images_for_test), colorama.Fore.RESET)
print(f"Number of TEST labels found:{colorama.Fore.CYAN}", len(labels_for_test), colorama.Fore.RESET)
print(f"Number of unique characters:{colorama.Fore.CYAN}", len(characters), colorama.Fore.RESET)
print(f"Characters present:{colorama.Fore.YELLOW}", characters, colorama.Fore.RESET)
tf.debugging.set_log_device_placement(True)

start_time = time.time()

"""
## Preprocessing
"""


# Splitting data into training and validation sets
x_train, x_valid, y_train, y_valid = split_data(np.array(images), np.array(labels),
        test_images=np.array(images_for_test), test_labels=np.array(labels_for_test),
                                                train_size=train_ratio)

# Create `Dataset` objects
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

## Visualize the data
###_, ax = plt.subplots(4, 4, figsize=(10, 5))
###for batch in train_dataset.take(1):
###    images = batch["image"]
###    labels = batch["label"]
###    for i in range(16):
###        img = (images[i] * 255).numpy().astype("uint8")
###        label = tf.strings.reduce_join(num_to_char(labels[i])).numpy().decode("utf-8")
###        ax[i // 4, i % 4].imshow(img[:, :, 0].T)  # .imshow(img.T, cmap="gray")
###        ax[i // 4, i % 4].set_title(label)
###        ax[i // 4, i % 4].axis("off")
###plt.show()

# Get the model
model = build_model()
if Path(MODEL_FNAME).is_dir():
    print('MODEL EXISTING! LOADING AND FITTING!!')
    model.load_weights(MODEL_FNAME)
    #model = keras.models.load_model(MODEL_FNAME)
else:
    print('EXISTING MODEL NOT FOUND! CREATING!')





model.summary()

## Training

# Add early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    callbacks=[early_stopping],
)

# Get the prediction model by extracting layers till the output layer
prediction_model = keras.models.Model(
    model.get_layer(name="image").input,
    model.get_layer(name="dense2").output
)
prediction_model.summary()

#  Let's check results on some validation samples
for batch in validation_dataset.take(1):
    batch_images = batch["image"]
    batch_labels = batch["label"]

    preds = prediction_model.predict(batch_images)
    pred_texts, percent = decode_batch_predictions(preds)

    orig_texts = []
    for label in batch_labels:
        label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
        orig_texts.append(label)

###    _, ax = plt.subplots(4, 4, figsize=(15, 5))
###    for i in range(len(pred_texts)):
###        img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
###        img = img.T
###        title = f"{round(percent * 100, 2)}% {pred_texts[i]}; {orig_texts[i]}"
###        ax[i // 4, i % 4].imshow(img, cmap="gray")
###        ax[i // 4, i % 4].set_title(title)
###        ax[i // 4, i % 4].axis("off")
###plt.show()

# Save the models

model.save(MODEL_FNAME_TRANING)
prediction_model.save(MODEL_FNAME)
from to_onnx import convertToOnnx
convertToOnnx()
print(f"{colorama.Fore.GREEN}Model + ONNX successfully saved!{colorama.Fore.RESET}",
      f'It took {colorama.Fore.CYAN}{(time.time() - start_time) / 60}{colorama.Fore.RESET} minutes.', sep='\n')
from onnx_tester import runOnnxTest
print('!!!!!!! RUNNING ONNX TEST !!!!!')
runOnnxTest()
print('!!!!!!! ALL STOPPED !!!!!')