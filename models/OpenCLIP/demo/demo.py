import os
import argparse
import time
import numpy as np
from PIL import Image
from transformers import CLIPProcessor
import demo


class OpenClip:
    def __init__(self, args):
        self.processor = CLIPProcessor.from_pretrained(args.config_path)
        self.max_length = self.processor.tokenizer.model_max_length
        self.model = demo.OpenClip()
        self.model.init(args.model_path)
        print("Init done!")

    def pipeline(self):
        while True:
            self.image_path = input("\nImage: ")
            if not self.image_path:
                print("Please enter the correct image path!")
                continue
            if self.image_path in ["exit","quit"]:
                break
            image = Image.open(self.image_path)

            self.text = input("\nDescription: ")
            if len(self.text) == 0:
                print("Please enter the correct text and split them by ';'")
                continue
            if self.text.split(";")[0] in ["exit","quit"]:
                break

            inputs = self.processor(text=self.text.split(";"),
                       images=image, return_tensors="np", padding=True)
            input_ids, attention_mask, pixel_values = inputs[
                'input_ids'], inputs['attention_mask'], inputs['pixel_values']
            ex_len = self.max_length - input_ids.shape[-1]
            pad_input = np.pad(input_ids, ((0, 0), (0, ex_len)), mode='constant', constant_values=0)
            pad_mask = np.pad(attention_mask, ((0, 0), (0, ex_len)), mode='constant', constant_values=0)

            print("\nAnswer: ")
            self.inference(pad_input.flatten().tolist(), pad_mask.flatten().tolist(), pixel_values.flatten().tolist())
    
    def inference(self, pad_input, pad_mask, pixel_values):
        start_time = time.time()
        probs = self.model.forward(pad_input, pad_mask, pixel_values)
        end_time = time.time()
        print(probs)
        # print(' '.join([str(item) for item in probs]))
        print("\nTime: {:.3f} s".format(end_time - start_time))
    

def main(args):
    model = OpenClip(args)
    model.pipeline()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='Path to the bmodel file.')
    parser.add_argument('--config_path', type=str, help='Path to the config file.')
    args = parser.parse_args()
    main(args)
            