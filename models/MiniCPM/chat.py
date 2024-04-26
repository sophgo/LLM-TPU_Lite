from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys
import re

path = '/workspace/MiniCPM-2B-sft-bf16'

def main():
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    history = []
    print("Question:")
    role = "user"
    query = sys.stdin.readline().strip()
    while query.lower() not in ['exit','quit']:
        history.append({"role": role, "content": query})
        history_str = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=False)
        print("Answer:")
        inputs = tokenizer(history_str, return_tensors='pt')
        outputs = model.generate(**inputs, max_new_tokens=2048)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):-1]
        response = tokenizer.decode(outputs)
        pattern = re.compile(r".*?(?=<AI>|<用户>)", re.DOTALL)
        matches = pattern.findall(response)
        history.append({"role": "assistant", "content": response})
        if len(matches) > 0:
            response = matches[0]
        print(response)
        print("Question:")
        query = sys.stdin.readline().strip()


if __name__ == '__main__':
    main()