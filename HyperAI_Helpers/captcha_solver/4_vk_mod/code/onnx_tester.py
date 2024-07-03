def runOnnxTest():
    import time
    import cv2
    import numpy as np
    import onnxruntime
    import requests
    from matplotlib import pyplot as plt, image as mpimg
    from utils.config import characters, img_height, img_width, img_type, data_dir_test, max_length, OUTPUT_ONNX, transpose_perm
    from utils.tester import mistakes_analyzer

    real_tests = False

    data_dir_test = data_dir_test.glob(img_type)
    sess = onnxruntime.InferenceSession(OUTPUT_ONNX) # r"../out.model.onnx")
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


    def solve(file_name: str):
        with open(file_name, 'rb') as F:
            data_bytes = F.read()
        img = decode_img(data_bytes)
        #print(img)
        pred_onx = sess.run(None, {name: img})[0]
        ans = get_result(pred_onx)
        return ans


    if True:
        mistakes = []
        total = 0
        correct = 0
        for file in data_dir_test:
            ans = file.name.split(".")[0].split('_')[0]
            solved_ans, solved_accuracy = solve(str(file))
            if solved_ans == ans:
                correct += 1
                mistakes.append((ans, "! ! ! !", solved_accuracy,1.0)) #debug
            else:
                mistake_num = 0
                for i,digit in enumerate(ans):
                    if i+1<=len(solved_ans):
                        if digit == solved_ans[i]:
                            mistake_num+=1
                mistake_ratio = mistake_num / len(ans)
                mistakes.append((ans, solved_ans, solved_accuracy,mistake_ratio))
            total += 1
            if total % 100 == 0:
                print(f"Success: {correct / total:.2%}", end='\r')
        print(f"\n\nSuccess: {correct}/{total} accuracy = {correct / total:.2%}", end='\n\n')
        for i,mistake in enumerate(mistakes):
            print(i+1,'\tотгадывание pred =',mistake[1],'\tsolve =',mistake[0], '\tуверен =',round(mistake[2],2),'\tсовпадение =',round(mistake[3],2))
if __name__ == "__main__":
    runOnnxTest()
