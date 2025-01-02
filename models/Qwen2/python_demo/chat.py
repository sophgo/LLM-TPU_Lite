import time
import argparse
from transformers import AutoTokenizer
import chat

class Engine:
    def __init__(self, args):
        # devid
        self.devices = [int(d) for d in args.devid.split(",")]

        # load tokenizer
        print("Load " + args.tokenizer_path + " ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path, trust_remote_code=True
        )

        # warm up
        self.tokenizer.decode([0])

        # preprocess parameters, such as prompt & tokenizer
        self.system_prompt = "You are a helpful assistant."
        self.history = [{"role": "system", "content": self.system_prompt}]
        self.EOS = self.tokenizer.eos_token_id
        self.enable_history = args.enable_history

        # load model
        self.model = chat.Qwen2()
        self.model.init(self.devices, args.model_path)

        # warm up
        self.tokenizer.decode([0])

    def clear(self):
        self.history = [{"role": "system", "content": self.system_prompt}]

    def update_history(self):
        if self.model.token_length >= self.model.SEQLEN:
            print("... (reach the maximal length)", flush=True, end="")
            self.history = [{"role": "system", "content": self.system_prompt}]
        else:
            self.history.append({"role": "assistant", "content": self.answer_cur})

    def encode_tokens(self):
        self.history.append({"role": "user", "content": self.input_str})
        text = self.tokenizer.apply_chat_template(
            self.history, tokenize=False, add_generation_prompt=True
        )
        tokens = self.tokenizer(text).input_ids
        return tokens

    def chat(self):
        print(
            """\n=================================================================
1. If you want to quit, please enter one of [q, quit, exit]
2. To create a new chat session, please enter one of [clear, new]
================================================================="""
        )
        while True:
            self.input_str = input("\nQuestion: ")
            if self.input_str in ["exit", "q", "quit"]:
                break
            elif self.input_str in ["clear", "new"]:
                self.clear()
            else:
                tokens = self.encode_tokens()
                if not tokens:
                    print("Sorry: your question is empty!!")
                    return
                if len(tokens) > self.model.SEQLEN:
                    print(
                        "The maximum question length should be shorter than {} but we get {} instead.".format(
                            self.model.SEQLEN, len(tokens)
                        )
                    )
                    return

                print("\nAnswer: ", end="")
                self.stream_answer(tokens)

    def stream_answer(self, tokens):
        tok_num = 1
        self.answer_cur = ""

        # First token
        first_start = time.time()
        token = self.model.forward_first(tokens)
        first_end = time.time()

        # Following tokens
        while token != self.EOS and self.model.token_length < self.model.SEQLEN:
            word = self.tokenizer.decode(token, skip_special_tokens=True)
            self.answer_cur += word
            print(word, flush=True, end='')
            token = self.model.forward_next()
            tok_num += 1

        # counting time
        next_end = time.time()
        first_duration = first_end - first_start
        next_duration = next_end - first_end
        tps = tok_num / next_duration

        if self.enable_history:
            self.update_history()
        else:
            self.clear()

        print()
        print(f"FTL: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} token/s")

def main(args):
    engine = Engine(args)
    engine.chat()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--devid', type=str, default='0', help='Device ID to use.')
    parser.add_argument('-m', '--model_path', type=str, help='Path to the bmodel file.')
    parser.add_argument('-t', '--tokenizer_path', type=str, default='../support/token_config', help='Path to the tokenizer file.')
    parser.add_argument('--enable_history', action='store_true', help="if set, enables storing of history memory")
    args = parser.parse_args()
    main(args)