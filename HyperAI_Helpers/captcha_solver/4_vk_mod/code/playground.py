import os
import queue
import random
from pathlib import Path
import time
import multiprocessing, threading

OneObjQueue = multiprocessing.Queue()

if __name__ == '__main__':
    OneObjQueue.put('645747cho63')
    #OneObjQueue.put('cho')
    #print(OneObjQueue.get(timeout=1))
    def worker():
        while True:
            try:
                print("a",OneObjQueue.get(block=False))
            except queue.Empty:
                print('no')
            time.sleep(0.5)
    threading.Thread(target=worker).start()
    time.sleep(5)
    print('PUT!')
    OneObjQueue.put('645747cho63')
    time.sleep(10)
    print('EXIT')
#max_length = 5
#
#data_dir_test = Path("images/test") #Path("../../images/test")
#img_type = "*.png" # "*.jpeg"  # "*.png"
#
#images = list(map(str, [i for i in data_dir_test.glob(img_type)]))
#images = sorted(images, key=lambda *h: random.random())
#labels = [
#    #img.split(os.path.sep)[-1].split('.')[0].ljust(max_length)[:max_length]
#    img.split(os.path.sep)[-1].split('_')[0].split('.png')[0].ljust(max_length)[:max_length]
#    #img.split(os.path.sep)[-1].split('.')[0].ljust(max_length)[:max_length]
#    for img in images
#]
#print(labels)