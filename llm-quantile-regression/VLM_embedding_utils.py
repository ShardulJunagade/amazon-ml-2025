"""Utilities for extracting image/text embeddings using Qwen2.5 VL 7B.

This module provides helpers to load the Qwen2.5 VL 7B model and compute
embeddings for text, images, or paired inputs. It also includes utilities to compute
similarity between embeddings.

Default target model: "Qwen/Qwen2.5-VL-7B-Instruct"
"""

from typing import List, Optional, Tuple, Union
import os
import matplotlib.pyplot as plt
import logging
import requests

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_VLM_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


def get_device(force_cpu: bool = False) -> torch.device:
    if not force_cpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_vlm(
    model_id: str = DEFAULT_VLM_ID, device: Optional[torch.device] = None
) -> Tuple[AutoProcessor, Qwen2_5_VLForConditionalGeneration, torch.device]:
    """Load processor + model for Qwen2.5 VL 7B.

    Returns (model, processor, device).
    """
    if device is None:
        device = get_device()

    logger.info("Loading processor for %s", model_id)
    processor = AutoProcessor.from_pretrained(model_id)

    logger.info("Loading model for %s", model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, dtype=torch.float16, attn_implementation="sdpa")
    model.to(device)
    
    print(model.device, ": device for model")
    model.eval()

    logger.info("Loaded model on device=%s with 16-bit precision", device)
    return model, processor, device


def generate_response(messages: List[dict], processor, model, max_new_tokens: int = 128) -> str:
    """Generate a response for a given set of messages using Qwen2.5 VL 7B."""
    # Prepare inputs using apply_chat_template
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt"
    ).to(model.device)

    # Debugging: Print inputs to verify
    logger.debug("Prepared inputs: %s", inputs)

    # Fallback: Check if input_ids are None
    if inputs.get("input_ids") is None:
        raise ValueError("Processor did not generate valid input_ids. Verify input preparation.")

    # Generate response
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0] if output_text else ""


def get_embeddings_from_dataset(dataset: List[dict], processor, model) -> List[torch.Tensor]:
    """Compute embeddings for image + text inputs from a dataset.

    Args:
        dataset (List[dict]): List of dictionaries containing "image_path" and "text" keys.
        processor: The processor for the Qwen2.5 VL model.
        model: The Qwen2.5 VL model.

    Returns:
        List[torch.Tensor]: List of embeddings for each input pair.
    """
    embeddings = []

    for entry in tqdm(dataset):
        image_path = entry.get("img_path")
        text = entry.get("text")

        if not image_path or not text:
            logger.warning("Skipping entry due to missing image or text: %s", entry)
            continue

        # Prepare inputs using apply_chat_template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image_path},
                    {"type": "text", "text": text}
                ]
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        # Debugging: Print inputs to verify
        # logger.debug("Prepared inputs: %s", inputs)

        # Fallback: Check if input_ids are None
        if inputs.get("input_ids") is None:
            logger.warning("Skipping entry due to invalid input_ids: %s", entry)
            continue

        # Compute embeddings
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            embedding = outputs.hidden_states[-1].mean(dim=1)  # Pooling: mean of last hidden state
            embeddings.append(embedding)
        #delete variables to free memory
        del inputs, outputs, embedding
        torch.cuda.empty_cache()

    return embeddings

# function to save embeddings to a file
def save_embeddings(embeddings: List[torch.Tensor], filename: str):
    """Save embeddings to a file."""
    torch.save(embeddings, filename)
    logger.info("Saved embeddings to %s", filename)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model, processor, device = load_vlm()

    # Example usage of get_embeddings_from_dataset
    dataset = [
        {"img_path": "../images/train_images/201054.jpg", "text": "Dark Chocolate Covered Cranberries by Its Delish, 5 lbs Bulk Bag | Gourmet Dark Chocolate Cranberries, Healthy Snacks Chocolate Fruit, USA Made, Vegan, Kosher Dark Chocolate Covered Cranberries | Various Sizes 5 lbs Bulk Bag Its Delish Brand | HIGHEST QUALITY Fresh and Tasty | ENJOY This Wonderful Treat As A Snack Or For Any Occasion | Certified Kosher OU"},
        {"img_path": "../images/train_images/18040.jpg", "text": "Black Walnut Yerba Mate Tea (Loose) (8 oz, ZIN: 547380) - 3 Pack Our Black Walnut Yerba Mate tea is a scrumptuous all-natural flavored Yerba Mate coffee substitute with Black Walnut Leaf, Walnuts and Rose Petals that will refresh you with its terrific taste! You will enjoy the earthy and acidity-free Black Walnut aroma and flavor again and again! - Ingredients: Yerba Mate, Black Walnut Leaf, Walnuts, Rose Petals and Natural Black Walnut Flavor. | No fillers. | Manufacturer: TerraVita | Size: 8 oz | Loose Leaf Yerba Mate Tea - Packed in a convenient upright pouch!"}
    ]
    embeddings = get_embeddings_from_dataset(dataset, processor, model)
    for i, emb in enumerate(embeddings):
        print(f"Embedding {i}: {emb}")
    
    # calculate cosine similarity between the two embeddings
    if len(embeddings) == 2:
        emb1 = embeddings[0].cpu().numpy()
        emb2 = embeddings[1].cpu().numpy()
        cos_sim = np.dot(emb1, emb2.T) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        print(f"Cosine similarity between embedding 0 and 1: {cos_sim.item()}")
        
        