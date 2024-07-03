import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils.config import *
from utils.config import transpose_perm
# Batch size for training and validation


def encode_img(img):
    # 1. Decode and convert to grayscale
    if img_type == '*.jpeg':
        img = tf.io.decode_jpeg(img, channels=3, dct_method='INTEGER_ACCURATE')
    #        img = tf.image.rgb_to_grayscale(img)
    else:
        img = tf.io.decode_png(img, channels=3)

    # 2. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 3. Resize to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    # 4. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=transpose_perm)
    return img
def encode_img_real(img):
    return {"image": encode_img(tf.io.read_file(img))}
def encode_single_sample(img_path, label):
    # Read image
    img = tf.io.read_file(img_path)
    # Preprocessing

    img = encode_img(img)
    # Map the characters in label to numbers
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    # Return a dict as our model is expecting two inputs
    return {"image": img, "label": label}
def decode_batch_predictions(pred):
    """    A utility function to decode the output of the network
    :param pred:
    :return: (str, int)
    """
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)
    #results, percent = results[0][0][:, :max_length],
    results = results[0][0][:, :max_length]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8").replace('[UNK]', '')
        output_text.append(res)
    if any(i for i in output_text if i not in characters):
        return output_text, 0
    return output_text, 1


# Mapping characters to integers
char_to_num = layers.StringLookup(vocabulary=characters, mask_token=None)
# Mapping integers back to original characters
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

