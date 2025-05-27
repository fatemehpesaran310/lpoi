from transformers import AutoTokenizer, AutoModelForVision2Seq, AutoProcessor
import torch
import json
import time
from peft import PeftModel, LoraModel
import os
import PIL
from PIL import Image
import requests
import requests.exceptions


def save_to_json(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


model_name = "HuggingFaceM4/idefics2-8b"
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    load_in_8bit=False,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

processor = AutoProcessor.from_pretrained(model_name, do_image_splitting=False)
model.load_adapter("./checkpoints/idefics2-8b-lpoi-list5-10k/final/")

with open('PATH_TO_AMBER_BENCHMARK/AMBER/data/query/query_all.json', 'r') as file:
    test_set = json.load(file)

dataset = []
total_num = len(test_set)
start = time.time()

def print_elapsed_time(start):
    elapsed = time.time() - start
    # format as 00:00:00
    print(f"=== Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(elapsed))} ===")


for item in test_set:
    user_message = item['query']
    image_path = item['image']
    image_path = f"PATH_TO_AMBER_BENCHMARK/AMBER/image/{image_path}"
    images = Image.open(image_path)
    data = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_message}]}]
    prompts = processor.apply_chat_template(data, add_generation_prompt=True)
    inputs = processor(prompts, images, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    generated_ids = model.generate(**inputs, max_new_tokens=500)
    response_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response_text)
    item['response'] = response_text
    # Delete the 'image' element after processing
    del item['image']
    del item['query']
    with open('./output/idefics2-8b-lpoi-list5-10k-amber.json', "w") as f:
        json.dump(test_set, f, indent=4)

# Preprocess the answer:

with open('./output/idefics2-8b-lpoi-list5-10k-amber.json', 'r') as file:
    data = json.load(file)

# Process the data starting from id 1004 or later
for entry in data:
    if int(entry['id']) >= 1004:  # Check if the ID is 1004 or later
        response = entry['response']
        if any(word in response for word in ['NO', 'No', 'no']):
            entry['response'] = 'No'
        elif any(word in response for word in ['YES', 'Yes', 'yes']):
            entry['response'] = 'Yes'

with open('./output/idefics2-8b-lpoi-list5-10k-amber.json', 'w') as file:
    json.dump(data, file, indent=4)    
print("Processing complete. The updated file has been saved as json")

