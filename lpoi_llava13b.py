
from datasets import load_dataset
import json
import os
import io  # Added import for io
import numpy as np

os.environ["WANDB__SERVICE_WAIT"] = "300"
# Load the dataset from the JSON file
dataset = load_dataset('json', data_files={'train': 'data/sample_10k_llava.json'})

from datasets import features
from transformers import AutoProcessor
import torch
from transformers import AutoModelForVision2Seq
from transformers import TrainingArguments
from peft import get_peft_model, LoraConfig
from datasets import features, load_dataset
from transformers import AutoModelForVision2Seq, AutoProcessor, LlavaForConditionalGeneration
import torch

from llava_13b_lpoi_5img import DPOTrainer
from lpoi_dpo_config import DPOConfig
from PIL import Image



# processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b", do_image_splitting=False)
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-13b-hf", do_image_splitting=False, revision='0fad4a3')

def format(example):
    # Print the current working directory
    print(f"Current working directory: {os.getcwd()}")

    # Print the full path of the image
    full_path = os.path.abspath(example["img_path"])
    negative1_path = full_path.replace("images", "ours_cmask_until_list5/ours_cmask_until25")
    negative2_path = full_path.replace("images", "ours_cmask_until_list5/ours_cmask_until50")
    negative3_path = full_path.replace("images", "ours_cmask_until_list5/ours_cmask_until75")
    negative4_path = full_path.replace("images", "ours_cmask_until_list5/ours_cmask_until100")
    print(f"Attempting to open image: {full_path}")
    if not os.path.exists(full_path):   
        print(f"Error: File does not exist: {full_path}")
        return example  # or handle the error as appropriate
    for negative_path in [negative1_path, negative2_path, negative3_path, negative4_path]:
        if not os.path.exists(negative_path):
            print(f"Error: File does not exist: {negative_path}")
            return example  # or handle the error as appropriate

    image = Image.open(full_path)
    # Convert the image to bytes
    byte_arr = io.BytesIO()
    image.save(byte_arr, format='PNG')  # Save as PNG or any other format
    byte_arr = byte_arr.getvalue()  # Get the byte data

    negative1_image = Image.open(negative1_path)
    negative1_byte_arr = io.BytesIO()
    negative1_image.save(negative1_byte_arr, format='PNG')  # Save as PNG or any other format
    negative1_byte_arr = negative1_byte_arr.getvalue()  # Get the byte data
    
    negative2_image = Image.open(negative2_path)
    negative2_byte_arr = io.BytesIO()
    negative2_image.save(negative2_byte_arr, format='PNG')  # Save as PNG or any other format
    negative2_byte_arr = negative2_byte_arr.getvalue()  # Get the byte data

    negative3_image = Image.open(negative3_path)
    negative3_byte_arr = io.BytesIO()
    negative3_image.save(negative3_byte_arr, format='PNG')  # Save as PNG or any other format
    negative3_byte_arr = negative3_byte_arr.getvalue()  # Get the byte data

    negative4_image = Image.open(negative4_path)
    negative4_byte_arr = io.BytesIO()
    negative4_image.save(negative4_byte_arr, format='PNG')  # Save as PNG or any other format
    negative4_byte_arr = negative4_byte_arr.getvalue()  # Get the byte data

    # Prepare the input for the chat template
    prompt = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": example["prompt"]}],
        },
    ]
    chosen = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example["chosen"]}],
        },
    ]
    rejected = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example["rejected"]}],
        },
    ]
    # Apply the chat template
    prompt = processor.apply_chat_template(prompt, tokenize=False)
    chosen = processor.apply_chat_template(chosen, tokenize=False)
    rejected = processor.apply_chat_template(rejected, tokenize=False)
    # Resize the image to ensure it fits within the maximum allowable
    # size of the processor to prevent OOM error
    max_size = max(processor.image_processor.crop_size["height"], processor.image_processor.crop_size["width"]) // 2
    # max_size = processor.image_processor.size["longest_edge"] // 2
    ## Change this to the cropped image
    image.thumbnail((max_size, max_size))
    negative1_image.thumbnail((max_size, max_size))
    negative2_image.thumbnail((max_size, max_size))
    negative3_image.thumbnail((max_size, max_size))
    negative4_image.thumbnail((max_size, max_size))
    # import ipdb; ipdb.set_trace()
    return {"images": [byte_arr], "negative1_images": [negative1_byte_arr], "negative2_images": [negative2_byte_arr], "negative3_images": [negative3_byte_arr], "negative4_images": [negative4_byte_arr], "prompt": prompt, "chosen": chosen, "rejected": rejected}

# Apply the formatting function to the dataset,
# remove columns to end up with only "images", "prompt", "chosen", "rejected" columns
dataset = dataset['train'].map(format)

# import pdb; pdb.set_trace()
f = dataset.features  # Or use 'validation' or 'test' depending on which split you need
f["images"] = features.Sequence(features.Image(decode=True))  # to avoid bytes
f["negative1_images"] = features.Sequence(features.Image(decode=True))  # to avoid bytes
f["negative2_images"] = features.Sequence(features.Image(decode=True))  # to avoid bytes
f["negative3_images"] = features.Sequence(features.Image(decode=True))  # to avoid bytes
f["negative4_images"] = features.Sequence(features.Image(decode=True))  # to avoid bytes

dataset = dataset.cast(f)

model = AutoModelForVision2Seq.from_pretrained("llava-hf/llava-1.5-13b-hf", torch_dtype=torch.float16, revision='0fad4a3', load_in_8bit=False)


peft_config = LoraConfig(target_modules="all-linear")
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


output_name = "llava-13b-lpoi-list5-10k"

# Train the model
training_args = DPOConfig(
    output_dir=output_name,
    gradient_checkpointing=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=64,
    num_train_epochs=1,
    dataset_num_proc=1,  # tokenization will use 32 processes
    dataloader_num_workers=4,  # data loading will use 32 workers
    logging_steps=50,
    save_strategy='epoch',
    add_image_dpo=True,
)

trainer = DPOTrainer(
    model,
    ref_model=None,  # not needed when using peft
    args=training_args,
    train_dataset=dataset,
    tokenizer=processor,
    peft_config=LoraConfig(target_modules="all-linear"),
)

trainer.train()
trainer.save_model(f"{output_name}/final")
