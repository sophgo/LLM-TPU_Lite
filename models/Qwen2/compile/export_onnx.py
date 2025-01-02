#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import os
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM
torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(description='export onnx')
parser.add_argument('-m', '--model_path', type=str, help='path to the torch model')
parser.add_argument('-s', '--seq_length', type=int, default=512, help="sequence length")
parser.add_argument('-d', '--device', type=str, choices=["cpu", "cuda"], default="cpu")

args = parser.parse_args()

model_path = args.model_path
folder = "./tmp/onnx"

device = torch.device(args.device)
if args.device == "cpu":
    dtype = torch.float
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    dtype = torch.bfloat16

origin_model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True,
    torch_dtype=dtype, device_map="auto").eval()

for param in origin_model.parameters():
    param.requires_grad = False

config = origin_model.config
transformer = origin_model.model
layers = transformer.layers

SEQ_LENGTH = args.seq_length
NUM_LAYERS = config.num_hidden_layers
HIDDEN_SIZE = config.hidden_size
NUM_ATTENTION_HEADS = config.num_attention_heads
NUM_KEY_VALUE_HEADS = config.num_key_value_heads
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS
VOCAB_SIZE = config.vocab_size

class Embedding(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_ids):
        out = transformer.embed_tokens(input_ids)
        return out.float()


class Block(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]

    def forward(self, hidden_states, position_ids, attention_mask):
        hidden_states, past_kv = self.layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True)
        present_k, present_v = past_kv
        return hidden_states.float(), present_k.float(), present_v.float()


class BlockCache(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]

    def forward(self, hidden_states, position_ids, attention_mask, past_k,
                past_v):
        hidden_states, past_kv = self.layer(
            hidden_states,
            past_key_value=(past_k, past_v),
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=True)
        present_k, present_v = past_kv
        return hidden_states.float(), present_k.float(), present_v.float()


class LmHeadWithTopk(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        hidden_states = transformer.norm(hidden_states)
        m_logits = origin_model.lm_head(hidden_states)
        _, token = torch.topk(m_logits.float(), 1)
        return token


def convert_block(layer_id):
    model = Block(layer_id)
    hidden_states = torch.randn(
        (1, SEQ_LENGTH, HIDDEN_SIZE)).to(dtype).to(device)
    position_ids = torch.tensor(
        [range(SEQ_LENGTH)], dtype=torch.long).to(device)
    attention_mask = torch.randn(
        (1, 1, SEQ_LENGTH, SEQ_LENGTH)).to(dtype).to(device)

    torch.onnx.export(
        model, (hidden_states, position_ids, attention_mask),
        f'{folder}/block_{layer_id}.onnx',
        verbose=False,
        input_names=['input_states', 'position_ids', 'attention_mask'],
        output_names=['hidden_states', 'past_k', 'past_v'],
        do_constant_folding=True,
        opset_version=15)


def convert_block_cache(layer_id):
    model = BlockCache(layer_id)
    hidden_states = torch.randn((1, 1, HIDDEN_SIZE)).to(dtype).to(device)
    position_ids = torch.tensor([range(1)], dtype=torch.long).to(device)
    attention_mask = torch.ones(
        (1, 1, 1, SEQ_LENGTH + 1)).to(dtype).to(device)
    past_k = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM)).to(dtype).to(device)
    past_v = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM)).to(dtype).to(device)

    torch.onnx.export(
        model, (hidden_states, position_ids, attention_mask, past_k, past_v),
        f'{folder}/block_cache_{layer_id}.onnx',
        verbose=False,
        input_names=[
            'input_states', 'position_ids', 'attention_mask', 'history_k',
            'history_v'
        ],
        output_names=['hidden_states', 'past_k', 'past_v'],
        do_constant_folding=True,
        opset_version=15)


def convert_embedding():
    model = Embedding()
    input_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.int32).to(device)
    module = torch.jit.trace(model.forward, input_ids)
    torch.jit.save(module, f'{folder}/embedding.pt')
    

def convert_lm_head():
    model = LmHeadWithTopk()
    hidden_states = torch.randn(1, 1, HIDDEN_SIZE).to(dtype).to(device)
    module = torch.jit.trace(model.forward, hidden_states)
    torch.jit.save(module, f'{folder}/lm_head.pt')


# create folder to store onnx
if not os.path.exists(folder):
    os.makedirs(folder)

# export models
print('Convert block & block_cache')
for i in tqdm(range(NUM_LAYERS)):
   convert_block(i)
   convert_block_cache(i)

print('Convert embedding')
convert_embedding()

print('Convert lm_head')
convert_lm_head()
