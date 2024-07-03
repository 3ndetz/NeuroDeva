def solve_captcha(data_bytes: bytes = None, file_name: str = None):
    import time
    import os
    import sys
    import cv2
    import numpy as np
    import onnxruntime


    real_tests = False

    thisfolder = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/')
    sys.path.insert(0, f'{thisfolder}/utils')
    from config import characters, img_height, img_width, img_type, max_length, transpose_perm, OUTPUT_ONNX

    sess = onnxruntime.InferenceSession(f"{thisfolder}/{OUTPUT_ONNX}") # r"../out.model.onnx")
    name = sess.get_inputs()[0].name
    def get_result(pred):
        """CTC decoder of the output tensor
        https://distill.pub/2017/ctc/
        https://en.wikipedia.org/wiki/Connectionist_temporal_classification
        :return string, float
        """
        accuracy = 1
        last = None
        ans = []
        # pred - 3d tensor, we need 2d array - first element
        for item in pred[0]:
            # get index of element with max accuracy
            char_ind = item.argmax()
            # ignore duplicates and special characters
            if char_ind != last and char_ind != 0 and char_ind != len(characters)+1:
                # this element is a character - append it to answer
                ans.append(characters[char_ind - 1])
                # Get accuracy for current character and multiply global accuracy by it
                accuracy *= item[char_ind]
            last = char_ind

        answ = "".join(ans)[:max_length]
        return answ, accuracy


    def decode_img(data_bytes: bytes):
        # same actions, as for tensorflow
        image = cv2.imdecode(np.asarray(bytearray(data_bytes), dtype=np.uint8), 1)
        image: "np.ndarray" = image.astype(np.float32) / 255.
        if image.shape != (img_height, img_width, 3):
            image = cv2.resize(image, (img_width, img_height))
        image = image.transpose(transpose_perm)
        #  Creating tensor ( adding 4d dimension )
        image = np.array([image])
        return image


    def solve(data_bytes: bytes=None, file_name=None):
        if file_name:
            with open(file_name, 'rb') as F:
                data_bytes = F.read()
        if data_bytes is None:
            print('[CAPTCHA RESOLVER NN] ПУСТОТА ВМЕСТО БАЙТОВ!')
            return None
        img = decode_img(data_bytes)
        #print(img)
        pred_onx = sess.run(None, {name: img})[0]
        ans = get_result(pred_onx)
        return ans
    result = solve(data_bytes,file_name)
    return {"result":result[0], "predict":result[1]}
if __name__ == "__main__":
    import os
    thisfolder = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/')
    print('RESULT =',solve_captcha(file_name=f"{thisfolder}/images/test/9408.png"))
    #print('RESULT =',solve_captcha())
