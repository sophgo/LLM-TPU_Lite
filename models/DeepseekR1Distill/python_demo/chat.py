import time
import argparse
from transformers import AutoTokenizer

import chat

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
        print("Load " + args.tokenizer_path + " ...")
        self.sp = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        self.EOS = self.sp.eos_token_id
        # self.EOS = [self.sp.convert_tokens_to_ids("<|end|>"), self.sp.convert_tokens_to_ids("<|im_end|>")]

        # load model
        devices = [int(d) for d in args.devid.split(",")]
        self.model = chat.Qwen()
        self.model.init(devices, self.sp.eos_token_id, args.model_path)

        # warm up
        self.sp.decode([0])
        print("Done!")

    def chat(self):
        # Stop Chatting with "exit" input
        while True:
            self.input_str = input("\nQuestion: ")
            if self.input_str in ["exit","quit"]:
                break

            # tokens_with_template = self.generate_tokens(self.input_str)
            self.messages.append({"role":"user","content":self.input_str})
            text = self.sp.apply_chat_template(
                self.messages,
                tokenize=False,
                add_generation_prompt=True
            )
            tokens = self.sp(text).input_ids

            print("\nAnswer: ")
            self.stream_answer(tokens)

            # if you want to use non-stream answer
            # res = self.model.answer(tokens)
            # print(self.sp.decode(res))

    def stream_answer(self, tokens):
        tok_num = 0
        self.answer_cur = ""

        if not tokens:
            print("Sorry: your question is too wierd!!")
            return
        if self.token_length > self.model.SEQLEN:
            print("The maximum question length should be shorter than {} but we get {} instead.".format(self.model.SEQLEN, self.token_length))
            return

        # First token
        first_start = time.time()
        token = self.forward_first(tokens)
        first_end = time.time()

        # Following tokens
        full_word_tokens = []
        while token != self.EOS and self.token_length < self.model.SEQLEN:
            full_word_tokens.append(token)
            diff = self.sp.decode(full_word_tokens, skip_special_tokens=True)
            if "�" in diff:
                token = self.forward_next(token)
                tok_num += 1
                continue
            self.answer_cur += diff
            print(diff, flush=True, end='')
            if self.token_length < self.model.SEQLEN:
                self.token_length += 1
            token = self.forward_next(token)
            tok_num += 1
            full_word_tokens = []

        # counting time
        next_end = time.time()
        first_duration = first_end - first_start
        next_duration = next_end - first_end
        tps = tok_num / next_duration

        if self.token_length >= self.model.SEQLEN:
            print("... (reach the maximal length)", flush=True, end='')
            self.messages = [{"role": "system", "content": self.system_prompt}]
            self.messages.append({"role": "user", "content": self.input_str})
            self.messages.append({"role": "assistant", "content": self.answer_cur})
        else:
            self.messages.append({"role": "assistant", "content": self.answer_cur})

        print()
        print(f"FTL: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} token/s")

    def forward_first(self, tokens):
        if self.mode == "greedy":
            token = self.model.forward_first(tokens)
        elif self.mode == "sample":
            token = self.model.forward_first_with_topk(tokens, self.mode)
        return token

    def forward_next(self, token):
        if self.mode == "greedy":
            token = self.model.forward_next(token)
        elif self.mode == "sample":
            token = self.model.forward_next_with_topk(token, self.mode)
        return token

def main(args):
    engine = Engine(args)
    engine.chat()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--devid', type=str, default='0', help='Device ID to use.')
    parser.add_argument('--model_path', type=str, help='Path to the bmodel file.')
    parser.add_argument('--tokenizer_path', type=str, help='Path to the tokenizer file.')
    args = parser.parse_args()
    main(args)

