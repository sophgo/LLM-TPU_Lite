import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)

def main():
    path = "/workspace/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, device_map="cuda", torch_dtype="auto", trust_remote_code=True)
    history = []
    print("Question:")
    role = "user"
    query = sys.stdin.readline().strip()
    while query.lower() not in ['exit','quit']:
        history.append({"role": role, "content": query})
        history_str = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=False)
        print("Answer:")
        inputs = tokenizer(history_str, return_tensors='pt').to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=500)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):-1]
        response = tokenizer.decode(outputs)
        history.append({"role": "assistant", "content": response})
        print(response)
        print("Question:")
        query = sys.stdin.readline().strip()


if __name__ == '__main__':
    main()