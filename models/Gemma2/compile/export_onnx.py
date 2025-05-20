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
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(description='export onnx.')
parser.add_argument('--model_path', required=True,
                    type=str, help='path to the torch model.')

args = parser.parse_args()

model_path = args.model_path
folder = f"./tmp/onnx"

causal_model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True,
    torch_dtype=torch.bfloat16, device_map="auto").eval()

for param in causal_model.parameters():
    param.requires_grad = False

config = causal_model.config
gemma_model = causal_model.model
layers = gemma_model.layers

SEQ_LENGTH = config.max_position_embeddings
NUM_LAYERS = config.num_hidden_layers
HIDDEN_SIZE = config.hidden_size
NUM_ATTENTION_HEADS = config.num_attention_heads
HEAD_DIM = config.head_dim
KV_HEAD = config.num_key_value_heads

print(f'Layers: {NUM_LAYERS}\nHidden size: {HIDDEN_SIZE}\n')

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._rotary_pos_emb_cache = None
        self._seq_len_cached = 0
        self._ntk_alpha_cached = 1.0
        self._ntk_alpha_cached_list = [1.0]

    def update_rotary_pos_emb_cache(self, max_seq_len, offset=0, ntk_alpha=1.0):
        seqlen = max_seq_len + offset
        if seqlen > self._seq_len_cached or ntk_alpha != self._ntk_alpha_cached:
            base = self.base * ntk_alpha ** (self.dim / (self.dim - 2))
            self.inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2,
                         device=self.inv_freq.device).float() / self.dim)
            )
            self._seq_len_cached = max(2 * seqlen, 16)
            self._ntk_alpha_cached = ntk_alpha
            seq = torch.arange(self._seq_len_cached,
                               device=self.inv_freq.device)
            freqs = torch.outer(seq.type_as(self.inv_freq), self.inv_freq)

            emb = torch.cat((freqs, freqs), dim=-1)
            from einops import rearrange

            emb = rearrange(emb, "n d -> 1 n 1 d")

            cos, sin = emb.cos(), emb.sin()
            self._rotary_pos_emb_cache = [cos, sin]

    def forward(self, max_seq_len, offset=0, ntk_alpha=1.0):
        self.update_rotary_pos_emb_cache(max_seq_len, offset, ntk_alpha)
        cos, sin = self._rotary_pos_emb_cache
        return [cos[:, offset: offset + max_seq_len], sin[:, offset: offset + max_seq_len]]


class Embedding(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_ids):
        inputs_embeds = gemma_model.embed_tokens(input_ids)
        out = inputs_embeds * (HIDDEN_SIZE**0.5)
        return out.float()


class GemmaBlock(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]
        self.rotary_emb = RotaryEmbedding(HEAD_DIM)
        self.rotary = self.rotary_emb(SEQ_LENGTH)
        self.cos_emb = self.rotary[0].view(
            SEQ_LENGTH, HEAD_DIM).bfloat16().to("cuda")
        self.sin_emb = self.rotary[1].view(
            SEQ_LENGTH, HEAD_DIM).bfloat16().to("cuda")

    def forward(self, hidden_states, position_ids, attention_mask):
        cos_pos = self.cos_emb[position_ids]
        sin_pos = self.sin_emb[position_ids]
        outputs = self.layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=False,
            use_cache=True,
            rotary_pos_emb_list=(cos_pos, sin_pos)
        )
        hidden_states = outputs[0]
        present_k, present_v = outputs[1]
        return hidden_states.float(), present_k.float(), present_v.float()


class GemmaBlockCache(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]
        self.rotary_emb = RotaryEmbedding(HEAD_DIM)
        self.rotary = self.rotary_emb(SEQ_LENGTH)
        self.cos_emb = self.rotary[0].view(
            SEQ_LENGTH, HEAD_DIM).bfloat16().to("cuda")
        self.sin_emb = self.rotary[1].view(
            SEQ_LENGTH, HEAD_DIM).bfloat16().to("cuda")

    def forward(self, hidden_states, position_ids, attention_mask, past_k,
                past_v):
        cos_pos = self.cos_emb[position_ids]
        sin_pos = self.sin_emb[position_ids]
        hidden_states, past_kv = self.layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=False,
            use_cache=True,
            layer_past=(past_k, past_v),
            rotary_pos_emb_list=(cos_pos, sin_pos)
        )
        present_k, present_v = past_kv
        return hidden_states.float(), present_k.float(), present_v.float()


class LmHead(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        hidden_states = gemma_model.norm(hidden_states)
        m_logits = causal_model.lm_head(hidden_states)
        _, token = torch.topk(m_logits.float(), 1)
        return token


def convert_block(layer_id):
    model = GemmaBlock(layer_id)
    hidden_states = torch.randn(
        (1, SEQ_LENGTH, HIDDEN_SIZE)).bfloat16().to("cuda")
    position_ids = torch.tensor(
        [range(SEQ_LENGTH)], dtype=torch.long).to("cuda")
    attention_mask = torch.randn(
        (1, 1, SEQ_LENGTH, SEQ_LENGTH)).bfloat16().to("cuda")

    torch.onnx.export(
        model, (hidden_states, position_ids, attention_mask),
        f'{folder}/block_{layer_id}.onnx',
        verbose=False,
        input_names=['input_states', 'position_ids', 'attention_mask'],
        output_names=['hidden_states', 'past_k', 'past_v'],
        do_constant_folding=True,
        opset_version=15)


def convert_block_cache(layer_id):
    model = GemmaBlockCache(layer_id)
    hidden_states = torch.randn((1, 1, HIDDEN_SIZE)).bfloat16().to("cuda")
    position_ids = torch.tensor([range(1)], dtype=torch.long).to("cuda")
    attention_mask = torch.ones(
        (1, 1, 1, SEQ_LENGTH + 1)).bfloat16().to("cuda")
    past_k = torch.randn(
        (1, SEQ_LENGTH, KV_HEAD, HEAD_DIM)).bfloat16().to("cuda")
    past_v = torch.randn(
        (1, SEQ_LENGTH, KV_HEAD, HEAD_DIM)).bfloat16().to("cuda")

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
    input_ids = torch.tensor([range(SEQ_LENGTH)]).to("cuda")
    module = torch.jit.trace(model.forward, input_ids)
    torch.jit.save(module, f'{folder}/embedding.pt')
    # torch.onnx.export(model, (input_ids),
    #                   f'{folder}/embedding.onnx',
    #                   verbose=False,
    #                   input_names=['input_ids'],
    #                   output_names=['input_embed'],
    #                   do_constant_folding=True,
    #                   opset_version=15)


def convert_lm_head():
    model = LmHead()
    input = torch.randn(1, HIDDEN_SIZE).bfloat16().to("cuda")
    module = torch.jit.trace(model.forward, input)
    torch.jit.save(module, f'{folder}/lm_head.pt')
    # torch.onnx.export(model, (input),
    #                   f'{folder}/lm_head.onnx',
    #                   verbose=False,
    #                   input_names=['hidden_states'],
    #                   output_names=['token'],
    #                   do_constant_folding=True,
    #                   opset_version=15)


# create folder to store onnx
if not os.path.exists(folder):
    os.makedirs(folder)


def test_net_with_mask():
    embed = Embedding()
    blocks = [GemmaBlock(i) for i in range(NUM_LAYERS)]
    block_kvs = [GemmaBlockCache(i) for i in range(NUM_LAYERS)]
    query = 'Write me a poem about Machine Learning.'
    print(query)
    ids = tokenizer.encode(query)
    print("input ids:{}".format(ids))
    token_len = len(ids)
    ids = ids + (SEQ_LENGTH - token_len) * [0]
    input_ids = torch.tensor(ids).view(SEQ_LENGTH).to("cuda")
    out = embed(input_ids).view(1, SEQ_LENGTH, HIDDEN_SIZE)
    position_ids = list(range(token_len)) + (SEQ_LENGTH - token_len) * [0]
    position_ids = torch.tensor([position_ids]).to("cuda")
    attention_mask = torch.ones((SEQ_LENGTH, SEQ_LENGTH)).float() * -10000.0
    for i in range(token_len):
        for j in range(token_len):
            if j <= i:
                attention_mask[i][j] = 0.0
    attention_mask = attention_mask.view(
        1, 1, SEQ_LENGTH, SEQ_LENGTH).to("cuda")
    k_cache = []
    v_cache = []
    for i in range(NUM_LAYERS):
        out, k, v = blocks[i](out.bfloat16(), position_ids,
                              attention_mask.bfloat16())
        k[:,:,token_len:,:] = 0
        v[:,:,token_len:,:] = 0
        k_cache.append(k)
        v_cache.append(v)
    out = out[:, token_len - 1:token_len].view(1, 1, HIDDEN_SIZE)
    lm = LmHead()
    token = lm(out.bfloat16()).view(1)
    out_ids = [int(token)]
    word = tokenizer.decode([int(token)])
    print(word, end="")
    while int(token) != tokenizer.eos_token_id and token_len < SEQ_LENGTH:
        token_len += 1
        input_ids = torch.tensor([token]).to("cuda")
        out = embed(input_ids).view(1, 1, HIDDEN_SIZE)
        position_ids = torch.tensor([[token_len - 1]]).to("cuda")
        attention_mask = torch.zeros(
            (1, 1, 1, SEQ_LENGTH + 1)).float().to("cuda")
        attention_mask[:, :, :, token_len-1:SEQ_LENGTH] = -10000.0
        for i in range(NUM_LAYERS):
            out, k, v = block_kvs[i](out.bfloat16(), position_ids,
                                     attention_mask.bfloat16(),
                                     k_cache[i].bfloat16(), v_cache[i].bfloat16())
            k_cache[i][:,token_len-1:token_len,:,:] = k[:,:,:,:]
            v_cache[i][:,token_len-1:token_len,:,:] = v[:,:,:,:]
        token = lm(out.bfloat16()).view(1)
        out_ids.append(int(token))
        word = tokenizer.decode([int(token)])
        print(word, end="")
    print("\noutput_ids:{}".format(out_ids))


# test_net_with_mask()

# export models
print(f'Convert block & block_cache')
for i in tqdm(range(NUM_LAYERS)):
    convert_block(i)
    convert_block_cache(i)

print(f'Convert embedding')
convert_embedding()

print(f'Convert lm_head')
convert_lm_head()
