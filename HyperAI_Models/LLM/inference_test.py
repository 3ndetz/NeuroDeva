import os
from datetime import datetime
import torch
print('torch name', torch.name)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def calcTime(time):
    return bcolors.OKGREEN + str(round((datetime.now() - time).total_seconds(), 1)) + bcolors.ENDC

def len_recursive(lst):
    if not lst:
        return 0
    return 1 + len_recursive(lst[1::2]) + len_recursive(lst[2::2])

t = datetime.now()

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration


from transformers.utils.import_utils import is_accelerate_available, is_bitsandbytes_available

print(is_accelerate_available(), is_bitsandbytes_available())
# 9k ckpt norm
# 12k ckpt хорошо работает нак SC1 и в диалоге

thisfolder = os.path.dirname(os.path.realpath(__file__))
t = datetime.now()
# MISTRAL

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
if False: # mistral model load
    from peft import PeftModel, PeftConfig
    mistral_model_id = "IlyaGusev/saiga_mistral_7b"
    mistral_model_cache_folder = thisfolder + "/variants/Mistral"

    config = PeftConfig.from_pretrained(mistral_model_id,
                                        cache_dir=mistral_model_cache_folder)
    mistral_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=mistral_model_cache_folder
    )
    mistral_model = PeftModel.from_pretrained(
        mistral_model,
        mistral_model_id,
        torch_dtype=torch.float16,
        cache_dir=mistral_model_cache_folder
    )
    mistral_model.eval()

    mistral_model.name = "mistral"

    mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model_id,
                                                      cache_dir=mistral_model_cache_folder, use_fast=False)
    mistral_generation_config = GenerationConfig.from_pretrained(mistral_model_id,
                                                                 cache_dir=mistral_model_cache_folder)
    print('==========mistral loaded, ' + calcTime(t))
    print(mistral_generation_config)

# INSTRUCT
t = datetime.now()
instruct_model_id = "SiberiaSoft/SiberianFredT5-instructor"
instruct_model_cache_folder = thisfolder + "/variants/SiberianInstructor"
DO_INSTRUCT_LOAD = True
instruct_tokenizer, instruct_model = None, None
if DO_INSTRUCT_LOAD:
    do_flan_load = True
    if do_flan_load:
        instruct_model_id = "Vikhrmodels/VikhrT5-3b"
        instruct_model_cache_folder = thisfolder + "/variants/VikhrFlanT5"

        instruct_tokenizer = AutoTokenizer.from_pretrained(instruct_model_id, trust_remote_code=True,
                                                           cache_dir=instruct_model_cache_folder)
        instruct_model = AutoModelForSeq2SeqLM.from_pretrained(instruct_model_id, device_map="auto", trust_remote_code=True,
                                                           cache_dir=instruct_model_cache_folder)
    else:
        instruct_tokenizer = AutoTokenizer.from_pretrained(instruct_model_id, trust_remote_code=True,
                                                           cache_dir=instruct_model_cache_folder)
        instruct_model = AutoModelForSeq2SeqLM.from_pretrained(instruct_model_id, trust_remote_code=True,
                                                               cache_dir=instruct_model_cache_folder,
                                                               torch_dtype=torch.bfloat16, device_map={'': 0})
    instruct_model.eval()
    print('==========instruct loaded, ' + calcTime(t))
    instruct_model.name = "INSTRUCT"

# DIALOG

dialog_model_id = "SiberiaSoft/SiberianPersonaFred-2"
dialog_model_cache_folder = thisfolder + "/variants/SiberianPersonaFred"

# dialog_tokenizer = AutoTokenizer.from_pretrained(dialog_model_id, trust_remote_code=True,
#                                                 cache_dir=dialog_model_cache_folder)
# dialog_model = AutoModelForSeq2SeqLM.from_pretrained(dialog_model_id, trust_remote_code=True,
#                                                     cache_dir=dialog_model_cache_folder, torch_dtype=torch.bfloat16,
#                                                     device_map={'': 0})
# dialog_model.eval()
# dialog_model.name = "PERSONA"
# print('==========DIALOG loaded, '+calcTime(t))

from transformers import GenerationConfig

# generation_config = GenerationConfig.from_pretrained("/media/denis/042CD5B7300C3479/restoration_anglicisms/training/rut5_asr/checkpoint-8000/")

import os
import time

_cached_stamp = 0
test_p_file = thisfolder + '/ppt_test.txt'


def check_file():
    global _cached_stamp

    stamp = os.stat(test_p_file).st_mtime
    if stamp != _cached_stamp:
        _cached_stamp = stamp
        if _cached_stamp != 0:
            return True
    return False


def get_test_file():
    def num(s):
        try:
            return int(s)
        except ValueError:
            return float(s)

    with open(test_p_file, encoding='utf8') as f:
        contents = f.readlines()
        output = "".join(contents)  # "\n".join(contents)
        output_mas = output.split('##_##')
        params_dict = {"temperature": 0.1, "min_length": 20, "max_new_tokens": 512, "top_p": 0.95, "top_k": 5,
                       "repetition_penalty": 1.03, "no_repeat_ngram_size": 2}
        if len(output_mas) > 1:
            for record in output_mas[1].replace('\n', '').split(', '):
                record = record.split('=')
                if len(record) > 1:
                    try:
                        params_dict[record[0]] = num(record[1])
                    except BaseException as err:
                        print('err reading record', err)
    return {"prompt": output_mas[0][0:-1], "params": params_dict}


def generate(prompt, model, tokenizer, type="dialog", params=None, return_data=True):
    t = datetime.now()
    starting_seq = ''
    ending_seq = ''
    gen_config = None

    if model.name != "mistral":
        starting_seq = '<SC6>'
        ending_seq = '<extra_id_0>'
    else:
        global mistral_generation_config
        gen_config = mistral_generation_config
    if type == "file":
        prepared_input = starting_seq + prompt
    else:
        prepared_input = starting_seq + prompt + '\nОтвет: ' + ending_seq
    if params is None:
        params = {"temperature": 0.1, "min_length": 20, "max_new_tokens": 512, "top_p": 0.95, "top_k": 5,
                  "repetition_penalty": 1.03,
                  "no_repeat_ngram_size": 2}

    if return_data:
        params_text = ""
        if params.get("show_params", 0):
            params_text = ": " + str(params)
        print('<<====================<< (data' + params_text + ')\n', prepared_input)
    data = tokenizer(prepared_input, return_tensors="pt")
    inp_len = len(data)
    data = {k: v.to(model.device) for k, v in data.items()}
    if gen_config:
        output_ids = model.generate(
            **data, generation_config=gen_config
        )[0]
        output_ids = output_ids[len(data["input_ids"][0]):]
    else:
        output_ids = model.generate(
            **data, do_sample=True, temperature=params["temperature"], max_new_tokens=params["max_new_tokens"],
            top_p=params["top_p"], top_k=params["top_k"], repetition_penalty=params["repetition_penalty"],
            no_repeat_ngram_size=params["no_repeat_ngram_size"], min_length=params["min_length"]
            # generation_config=generation_config
        )[0]
    out = tokenizer.decode(output_ids, skip_special_tokens=True)
    out = out.replace("<s>", "").replace("</s>", "").replace("<pad>", "").replace("<extra_id_0>", "")
    print(
        ">>====================>> (model: " + model.name + "), ",calcTime(t)+'s,', f'i/o=[{str(len(data["input_ids"][0]))}/{str(len(output_ids))}]')  # + ", time: "+calcTime(t)+f", usage i/o: {str(inp_len)}/{str(len(output_ids))})")
    return out


print('========== ALL READY, ' + calcTime(t))
while 1:  # file mode
    while not check_file():
        time.sleep(0.1)
    file_data = get_test_file()
    # print(generate(file_data["prompt"], mistral_model, mistral_tokenizer, type="file", params=file_data["params"]))
    print(generate(file_data["prompt"], instruct_model, instruct_tokenizer, type="file", params=file_data["params"]))
    # print(generate(file_data["prompt"], dialog_model, dialog_tokenizer, type="file", params=file_data["params"], return_data=False))
# while 1: #standart dialog mode
#
#  print(generate(input(":> ")))
