# -*- coding: utf-8 -*-
import torch
import gc
import os
import datetime
import traceback
import contextlib
from typing import Optional, Dict, List
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    GenerationConfig, 
    StoppingCriteria, 
    StoppingCriteriaList
)

def find_repeating_tokens(sample: list, check: list) -> bool:
    if len(check) > 10 and len(sample) > 10:
        for k, token in enumerate(check):
            if k >= 9:
                check_word = [check[k - 9:k + 1]]
                for i, sample_token in enumerate(sample):
                    if i >= 9:
                        sample_word = [sample[i - 9:i + 1]]
                        if check_word == sample_word:
                            return True
        return False
    return False

def calc_time(start_time: datetime.datetime) -> str:
    return f"\033[92m{str((datetime.datetime.now() - start_time).total_seconds())}\033[0m"

def cut_spaces(text: str) -> str:
    result = []
    prev_was_space = False
    for char in text:
        if char == ' ':
            if not prev_was_space:
                result.append(char)
            prev_was_space = True
        else:
            result.append(char)
            prev_was_space = False
    return ''.join(result).strip()

def get_cmd(text: str, tip: str = "emo") -> Dict[str, str]:
    result = ""
    if tip == "emo":
        cmdlist = "агрессия, скука, усталость, интерес, смущение, счастье, веселье, страх".split(', ')
        brackets = ['[', ']']
    else: 
        cmdlist = "бан, издевайся, попрыгай, смейся, кричи, убегай".split(', ')
        brackets = ['<', '>']

    lbracket_idx = text.find(brackets[0]) + 1
    rbracket_idx = text.rfind(brackets[1])

    if lbracket_idx > 0 and rbracket_idx > 0:
        emotion_container = text[lbracket_idx:rbracket_idx]
        for command in cmdlist:
            if command in emotion_container:
                result = command
                break
        text = text[:lbracket_idx - 1] + text[rbracket_idx + 1:]

    return {"cmd": result, "cut": text}

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids: list, keywords_words_ids: list, sample: list, control_out: list):
        self.i = 0
        self.control_out = control_out
        self.keywords = keywords_ids
        self.words = keywords_words_ids
        self.sample = sample

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        self.i += 1
        if input_ids[0][-1].item() in self.keywords:
            self.control_out.append('symbol')
            print('[STOP CRITERIA] Early stopping triggered! (by symbol)')
            return True

        if len(input_ids[0]) > 1:
            if [input_ids[0][-1].item(), input_ids[0][-2].item()] in self.words:
                self.control_out.append('word')
                print('[STOP CRITERIA] Early stopping triggered!')
                return True

        if self.i > 5 and self.i % 5 == 0:
            if find_repeating_tokens(self.sample, input_ids[0].tolist()):
                self.control_out.append('repeat')
                print('[STOP CRITERIA] Early stopping REPEAT FOUND')
                return True
        return False

class FredT5:
    def __init__(self):
        self.base_path = Path(os.path.dirname(os.path.realpath(__file__)))
        self.model_paths = {
            'instruct': {
                'id': 'SiberiaSoft/SiberianFredT5-instructor',
                'localPath': '/variants/SiberianInstructor'
            },
            'dialog': {
                'id': 'SiberiaSoft/SiberianPersonaFred-2',
                'localPath': '/variants/SiberianPersonaFred'
            }
        }
        
        self.tokenizer = None
        self.model = None
        self.initialized = False
        self.last_tokens_used = 0
        
        self.cuda_enabled = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_enabled else "cpu")
        self.max_model_memory = int(torch.cuda.mem_get_info()[0] / 1024 ** 3) - 2 if self.cuda_enabled else 18
        self.model_data_type = torch.bfloat16
        
        self.autocast = torch.cuda.amp.autocast if self.cuda_enabled else self._dummy_context

    @contextlib.contextmanager
    def _dummy_context(self, *args, **kwargs):
        yield

    def initialize(self, force_load: bool = False) -> None:
        if self.initialized and not force_load:
            return

        try:
            t = datetime.datetime.now()
            print(f'\n=== Loading FredT5 model on {self.device} ===\n')

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_paths["instruct"]["id"],
                cache_dir=self.base_path / self.model_paths["instruct"]["localPath"]
            )

            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_paths["instruct"]["id"],
                cache_dir=self.base_path / self.model_paths["instruct"]["localPath"],
                max_memory={0: f'{self.max_model_memory//2}GB'},
                torch_dtype=self.model_data_type,
                device_map={'': 0}
            )

            self.model.eval()
            self.initialized = True
            print(f'\n=== Model loading completed ({calc_time(t)}s) ===\n')

        except Exception as e:
            print(f"Failed to initialize model: {str(e)}")
            raise

    def _generate(self, model, input_ids, generation_config, stop_criteria):
        print('[LLM DEBUG MEM] BEFORE GENERATION:', torch.cuda.memory_allocated())
        
        if torch.cuda.memory_allocated() > 4000000000:
            torch.cuda.empty_cache()
            print('[LLM DEBUG MEM] Cache cleared')

        with torch.inference_mode():
            with self.autocast(enabled=True, dtype=self.model_data_type):
                with torch.no_grad():
                    result = model.generate(
                        input_ids,
                        generation_config=generation_config,
                        stopping_criteria=StoppingCriteriaList([stop_criteria])
                    )
                    print('[LLM DEBUG MEM] AFTER GENERATION:', torch.cuda.memory_allocated())
                    return result

    def _setup_stopping_criteria(self, repeat_danger_part: str) -> tuple:
        stop_symbols = ['}', '*', ']']
        stop_words = [['\n', '*'], ['\n', 'Q'], ['Q', ':']]
        stop_ids = [self.tokenizer.encode(w, add_special_tokens=False)[0] for w in stop_symbols]
        stop_ids_words = [[self.tokenizer.encode(w, add_special_tokens=False)[0] for w in word] for word in stop_words]
        
        sample_part = []
        if repeat_danger_part:
            sample_part = self.tokenizer.encode(repeat_danger_part, add_special_tokens=False)
            
        stopping_callback = []
        return KeywordsStoppingCriteria(stop_ids, stop_ids_words, sample_part, stopping_callback), stopping_callback

    async def generate_response(
        self,
        text: str,
        params: Optional[Dict] = None,
        repeat_danger_part: str = ''
    ) -> Dict:
        if not self.initialized:
            self.initialize()

        default_params = {
            "do_sample": True,
            "top_p": 0.95,
            "top_k": 5,
            "temperature": 0.2,
            "repetition_penalty": 1.03,
            "min_length": 10,
            "max_length": 150,
            "no_repeat_ngram_size": 2,
            "num_beams": 3,
            "max_time": 12.0,
            "tokens_offset": 0
        }
        params = {**default_params, **(params or {})}

        try:
            model_input = text + '<extra_id_0>'
            input_tokens = self.tokenizer.encode(model_input, add_special_tokens=False)
            input_ids = torch.tensor([input_tokens]).to(self.device)

            stop_criteria, stopping_callback = self._setup_stopping_criteria(repeat_danger_part)
            
            try:
                generation_config = GenerationConfig.from_pretrained(
                    self.model_paths["instruct"]["id"],
                    cache_dir=self.base_path / self.model_paths["instruct"]["localPath"]
                )
            except Exception as err:
                print(f'GenConfig not found: {err}')
                generation_config = GenerationConfig.from_dict({
                    "bos_token_id": 50256,
                    "eos_token_id": 50256,
                    "transformers_version": "4.27.1"
                })

            for key, value in params.items():
                if hasattr(generation_config, key):
                    setattr(generation_config, key, value)
            
            generation_config.eos_token_id = self.tokenizer.eos_token_id
            generation_config.early_stopping = True

            t = datetime.datetime.now()
            outputs = self._generate(self.model, input_ids, generation_config, stop_criteria)
            
            if len(outputs) > 0:
                self.last_tokens_used = len(outputs[0])
                output = outputs[0][1 + params["tokens_offset"]:]
                result = self.tokenizer.decode(output, skip_special_tokens=True)
                result = cut_spaces(result.replace('<extra_id_0>', '').replace('A:', '').strip())

                print(f'{calc_time(t)}s - Generation complete, tokens [I/O]=[{len(input_tokens)}/{self.last_tokens_used}]')

                cmd = get_cmd(result, tip="emo")
                emotion = cmd["cmd"]
                result = cmd["cut"]
                
                cmd = get_cmd(result, tip="cmd")
                command = cmd["cmd"]
                result = cmd["cut"]

                return {
                    "reply": result,
                    "emotion": emotion,
                    "command": command,
                    "tokens": self.last_tokens_used,
                    "stopped": stopping_callback[0] if stopping_callback else ""
                }

        except Exception as e:
            print(f"Error generating response: {str(e)}")
            traceback.print_exc()
            
        return {
            "reply": "",
            "emotion": "нет",
            "command": "нет",
            "tokens": 0,
            "stopped": ""
        }
