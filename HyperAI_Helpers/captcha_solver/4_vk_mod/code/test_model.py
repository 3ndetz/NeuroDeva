import os
import random
from utils.config import *
import numpy
import keras
#from keras.models import Functional
#from matplotlib import pyplot as plt
from tools import *


# Visualize data
#need_visualization = True

#model: Functional = keras.models.load_model(MODEL_FNAME)
model = keras.models.load_model(MODEL_FNAME)

images = list(map(str, [i for i in data_dir_test.glob(img_type)]))
images = sorted(images, key=lambda *h: random.random())
labels = [
    #img.split(os.path.sep)[-1].split('-')[0].split('.jpeg')[0].ljust(8)[:8]
    img.split(os.path.sep)[-1].split('_')[0].split('.')[0].ljust(max_length)[:max_length]
    for img in images
]
print("images, labels =",images,labels)
test_dataset = tf.data.Dataset.from_tensor_slices((numpy.array(images), numpy.array(labels)))
test_dataset = (
    test_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)
print("test_dataset =",test_dataset)
#print(images)
#def visualize():
#    _, ax = plt.subplots(4, 4, figsize=(15, 5))
#    for i in range(len(pred_texts)):
#        img = (image[i, :, :, :] * 255).numpy().transpose([1, 0, 2]).astype(np.uint8)
#        title = f" {pred_texts[i]} {'ok' if pred_texts[i] == orig_texts[i] else 'wrong'}"
#        ax[i // 4, i % 4].imshow(img)
#        ax[i // 4, i % 4].set_title(title)
#        ax[i // 4, i % 4].axis("off")
#    plt.show()


correct = 0
total = 0

for i in test_dataset.take(100):
    label, image = i['label'], i['image']
    #print('CSFDSG',i['label'])
    preds = model.call(image)
    pred_texts = decode_batch_predictions(preds)[0]
    #print('pred',pred_texts)
    orig_texts = [tf.strings.reduce_join(num_to_char(l)).numpy().decode("utf-8").replace('[UNK]', '') for l in label]
    #print('pred',orig_texts)
    #if need_visualization and total == 0:
    #    visualize()
    for i in range(len(pred_texts)):
        success = pred_texts[i] == orig_texts[i]
        if success:
            #print('CHO',pred_texts[i])
            correct += 1
        else:
            print('ОШИБКА:',orig_texts[i],'=',pred_texts[i])
    total += len(pred_texts)
    print(f"Progress {total/len(images):.2%}[{total}/{len(images)}]...", end='\r')
print("\n\n")
print(f"\n\nSuccess: {correct}/{total} accuracy = {correct / total:.2%}", end='\n\n')
