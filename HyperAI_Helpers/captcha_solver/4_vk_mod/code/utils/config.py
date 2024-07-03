# Path to the data directory
from pathlib import Path

import colorama
#https://github.com/imartemy1524/AITextCaptcha/tree/master
data_dir = Path("images/train")#Path("../../images/train")
data_dir_test = Path("images/test") #Path("../../images/test")
img_type = "*.png" # "*.jpeg"  # "*.png"
characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# Desired image dimensions
img_width = 70#130
img_height = 50#50
# Maximum length of any captcha in the dataset
max_length = 5

MODEL_FNAME_TRANING = "output/output.traning2.model"
MODEL_FNAME = "output/output2.model"
OUTPUT_ONNX = "output/out.model.onnx"

# Training config
batch_size = 16 # 16 #48 мое норм 77%
epochs = 100 # 400мое норм 77
early_stopping_patience = 20 #было 10 #40 мое норм 77%
train_ratio = 0.8

# соотношение TRAIN IMAGES к TEST IMAGES, 0.9 означает 0.9 для трейн и 0.1 для тест.
# Ставь 1.0 если для теста юзать только те что в папке тест

transpose_perm = [1, 0, 2]
# батч епохи патиенс

# 1
# 48 400 40 нормас 77% вроде 170 эпоха

# 2
#96 800 80
# loss 0.2-0.1 на 200 эпохе; 300 эпоха стабильно 0.1  потом меньше 0.1


# we all love colored output, aren't we?
colorama.init()
