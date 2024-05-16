#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

# python3 export_onnx.py --model_path /workspace/openai/clip-vit-base-patch32

import os
import torch
import torch.nn.functional as F
import argparse
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

parser = argparse.ArgumentParser(description='export onnx.')
parser.add_argument('--model_path', type=str, help='path to the torch model.')
args = parser.parse_args()
model_path = args.model_path
folder = f"./tmp/onnx"


class ClipModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_path)

    def forward(self, input_ids, attention_mask, pixel_values):
        outputs = self.model(input_ids, pixel_values, attention_mask)
        # this is the image-text similarity score
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        return probs


def convert_onnx():
    model = ClipModel()
    processor = CLIPProcessor.from_pretrained(model_path)
    image = Image.open("../000000039769.jpg")
    inputs = processor(text=["a photo of two cats", "a photo of a dog", "a photo of one cat"],
                       images=image, return_tensors="pt", padding=True)
    input_ids, attention_mask, pixel_values = inputs[
        'input_ids'], inputs['attention_mask'], inputs['pixel_values']
    max_length = processor.tokenizer.model_max_length
    ex_len = max_length - input_ids.shape[-1]
    pad_input = F.pad(input_ids, (0, ex_len))
    pad_mask = F.pad(attention_mask, (0, ex_len))
    torch.onnx.export(model, (pad_input, pad_mask, pixel_values),
                      f'{folder}/openclip.onnx',
                      verbose=False,
                      do_constant_folding=True,
                      input_names=['input_ids',
                                   'attention_mask', 'pixel_values'],
                      output_names=['probs'],
                      dynamic_axes={
                          'input_ids': {0: 'x'},
                          'attention_mask': {0: 'x'},
                          'probs': {1: 'x'}},
                      opset_version=15)


def test_net_with_mask():
    processor = CLIPProcessor.from_pretrained(model_path)
    image = Image.open("../000000039769.jpg")
    inputs = processor(text=["a photo of two cats", "a photo of a dog", "a photo of one cat"],
                       images=image, return_tensors="pt", padding=True)
    input_ids, attention_mask, pixel_values = inputs[
        'input_ids'], inputs['attention_mask'], inputs['pixel_values']
    max_length = processor.tokenizer.model_max_length
    ex_len = max_length - input_ids.shape[-1]
    pad_input = F.pad(input_ids, (0, ex_len))
    pad_mask = F.pad(attention_mask, (0, ex_len))
    model = ClipModel()
    probs = model(input_ids, attention_mask, pixel_values)
    print("probs:{}".format(probs))
    pad_probs = model(pad_input, pad_mask, pixel_values)
    print("probs with pad:{}".format(pad_probs))


# test_net_with_mask()

# create folder to store onnx
if not os.path.exists(folder):
    os.makedirs(folder)

# export models
print(f'Convert clip model')
convert_onnx()
