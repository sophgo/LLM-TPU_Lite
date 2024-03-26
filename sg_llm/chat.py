#!/usr/bin/env python3
import time
import argparse
from transformers import AutoTokenizer

from python import sg_llm

class Engine:
    def __init__(self, args):
        # preprocess parameters, such as prompt & tokenizer
        self.input_str = ""
        self.system_prompt = "You are a helpful assistant."
        self.messages = [{"role": "system", "content": self.system_prompt}]

        # model parameters
        self.token_length = 0

        # postprocess parameters
        self.mode = "greedy"

        # load tokenizer
        print("Load " + args.tokenizer + " ...")
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        self.EOS = self.tokenizer.eos_token_id

        self.model = sg_llm.sg_llm(args.model)
        self.MAX_SEQLEN = self.model.MAX_SEQLEN

        # warm up
        self.tokenizer.decode([0])

    def chat(self):
        # Stop Chatting with "exit" input
        while True:
            self.input_str = input("\nQuestion: ")
            if self.input_str in ["exit","quit"]:
                break

            # tokens_with_template = self.generate_tokens(self.input_str)
            self.messages.append({"role":"user","content":self.input_str})
            text = self.tokenizer.apply_chat_template(
                self.messages,
                tokenize=False,
                add_generation_prompt=True
            )
            tokens = self.tokenizer(text).input_ids

            print("\nAnswer: ")
            self.stream_answer(tokens)


    def stream_answer(self, tokens):
        tok_num = 0
        self.answer_cur = ""

        if not tokens:
            print("Sorry: your question is too wierd!!")
            return
        if self.token_length > self.MAX_SEQLEN:
            print("The maximum question length should be shorter than {} but we get {} instead.".format(self.MAX_SEQLEN, self.token_length))
            return

        # First token
        first_start = time.time()
        token = self.model.forward_first(tokens)
        first_end = time.time()

        # Following tokens
        while token != self.EOS and self.token_length < self.MAX_SEQLEN:
            diff = self.tokenizer.decode([token])
            self.answer_cur += diff
            print(diff, flush=True, end='')
            if self.token_length < self.MAX_SEQLEN:
                self.token_length += 1
            tok_num += 1
            token = self.model.forward_next(token)

        # counting time
        next_end = time.time()
        first_duration = first_end - first_start
        next_duration = next_end - first_end
        tps = tok_num / next_duration

        if self.token_length >= self.MAX_SEQLEN - 128:
            print("... (reach the maximal length)", flush=True, end='')
            self.messages = self.messages[0]
            self.messages.append({"role": "user", "content": self.input_str})
            self.messages.append({"role": "assistant", "content": self.answer_cur})
        else:
            self.messages.append({"role": "assistant", "content": self.answer_cur})

        print(f"\nFTL: {first_duration:.3f} s, TPS: {tps:.3f} token/s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Path to the bmodel file.')
    parser.add_argument('--tokenizer', type=str, help='Path to the tokenizer file.')
    args = parser.parse_args()
    engine = Engine(args)
    engine.chat()
