from .. import GECModel
import pathlib
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel, get_peft_model
import torch
import json

import os
import logging
import gc
import requests
import tarfile

device_map = {"": 0}

logger = logging.getLogger(__name__)


class LlamaGEC(GECModel):

    def __init__(self, device=None, model_path=None, max_length=512):
        self.base_dir = pathlib.Path(__file__).resolve().parent
        super(LlamaGEC, self).__init__(device=device, model_path=model_path)
        self.has_tokenizer_detokenizer = True
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.max_length = max_length
        
        # self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.bfloat16, quantization_config=self.bnb_config)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        self.model.push_to_hub("llama-2-gpt4-grammar")

        self.pipe = pipeline(task="text-generation", model=self.model, tokenizer=self.tokenizer, max_length=100)

    def predict(self, sents, num_return_sequences=1, num_beams=4, temperature=1.5):
        """
        Input sentences are NOT tokenized.
        """
        # Monitor GPU memory usage
        mem_free, mem_total = torch.cuda.mem_get_info()
        logger.info(f"GPU memory current usage: {mem_total/1024/1024} MiB total, {mem_free/1024/1024} MiB free")
        if mem_free < 1024:
            logger.info("GPU memory is running low. Clearing cache...")
            gc.collect()            
            torch.cuda.empty_cache()
            mem_free, mem_total = torch.cuda.mem_get_info()
            logger.info(f"GPU memory current usage: {mem_total/1024/1024} MiB total, {mem_free/1024/1024} MiB free")
            logger.info(f"GPU memory summary: {torch.cuda.memory_summary()}")
        try:
            tgt_text = []
            sents = [f"<s>[INST] Correct the grammar in the following sentence: {s} [/INST]" for s in sents]
            result = self.pipe(sents)
            for i in range(len(result)):
                output_text = result[i][0]['generated_text']
                output_text = output_text[output_text.index("[/INST]") + 7:]
                if "[/INST]" in output_text:
                    output_text = output_text[:output_text.index("[/INST]")]
                if "(" in output_text:
                    output_text = output_text[:output_text.index("(")]
                output_text = (output_text.split("\n")[0]).strip(" ")
                if ". " in output_text:
                    tgt_text.append(sents[i])
                else:
                    tgt_text.append(output_text)

        except Exception as e:
            tgt_text = []
            logger.error(f"Error in LlamaGEC.predict: {e}")
        gc.collect()        
        torch.cuda.empty_cache()
        print("LLAMA OUTPUT =>", tgt_text)
        return tgt_text