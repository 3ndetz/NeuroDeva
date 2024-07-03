#pip install cv2
#pip install onnxruntime

#from vk_captcha import VkCaptchaSolver
from captcha_solver.solver import VkCaptchaSolver

import requests
import random
import cv2
solver = VkCaptchaSolver(characters_param= ['z', 's', 'h', 'q', 'd', 'v', '2', '7', '8', 'x', 'y', '5', 'e', 'a', 'u', '4', 'k', 'n', 'm', 'c', 'p'])
print('all loaded!')
#img_bytes = requests.get(f"https://api.vk.com/captcha.php?sid={random.randint(0,10000000)}").content

def get_resized_image_bytes():
    im = cv2.imread('samples/captcha.jpg')
    im_resize = cv2.resize(im, (130, 50))

    is_success, im_buf_arr = cv2.imencode(".jpg", im_resize)
    byte_im = im_buf_arr.tobytes()
    return byte_im

def get_sample_vk_captcha_bytes():
    return requests.get(f"https://api.vk.com/captcha.php?sid={random.randint(0,10000000)}").content

def get_raw_image_bytes(): # не работает! нужен 130x50
    with open('lastcaptha.png', 'rb') as f:
        img_bytes = f.read()
    return img_bytes

print('loaded img')
answer, accuracy = solver.solve(bytes_data=get_resized_image_bytes())
accuracy = round((1.0-accuracy)*100, 4)
print('SOLVED! ans, acc =',answer,accuracy,'%')
exit()