from PIL import Image
import requests

from transformers import CLIPProcessor
from compile.files.clip.modeling_clip import CLIPModel

model = CLIPModel.from_pretrained("/workspace/openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("/workspace/openai/clip-vit-base-patch32")

image = Image.open("000000039769.jpg")

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

print("probs:[{}]".format(probs))