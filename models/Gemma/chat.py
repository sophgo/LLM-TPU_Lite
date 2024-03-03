#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import datetime
import math
import unittest
import torch
import random
import sys
import os

from transformers import AutoModelForCausalLM, AutoTokenizer

PWD = os.getcwd().replace('\\','/')

GEMMA_PATH = "{}/../../../gemma-2b-it".format(PWD)

print("Gemma path:{}".format(GEMMA_PATH))

if not os.path.exists(GEMMA_PATH):
    raise RuntimeError("Gemma not exist")

def get_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH,
                                              trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        GEMMA_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto").eval()
    return model, tokenizer

def main():
    model, tokenizer = get_model_and_tokenizer()
    print("Question:")
    s = sys.stdin.readline().strip()
    while s.lower() not in ['exit','quit']:
        print("Answer:")
        input_ids = tokenizer(s, return_tensors="pt").to("cuda")
        outputs = model.generate(**input_ids, max_new_tokens=2048)
        response = tokenizer.decode(outputs[0])
        print(response)
        print("Question:")
        s = sys.stdin.readline().strip()


if __name__ == '__main__':
    main()
