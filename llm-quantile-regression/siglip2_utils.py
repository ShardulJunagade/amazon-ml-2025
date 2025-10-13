import torch
import numpy as np
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image

def load_siglip2_model_and_processor(ckpt="google/siglip2-giant-opt-patch16-384"):
    """
    Load the SigLIP2 Giant model and processor.

    Args:
        ckpt (str): The checkpoint name or path.

    Returns:
        model: The loaded SigLIP2 model.
        processor: The corresponding processor.
    """
    model = AutoModel.from_pretrained(ckpt, device_map="auto").eval()
    processor = AutoProcessor.from_pretrained(ckpt)
    return model, processor

def embed_image(image_path, model, processor):
    """
    Generate embeddings for a single image using the SigLIP2 model.

    Args:
        image_path (str): Path to the image file.
        model: The SigLIP2 model.
        processor: The SigLIP2 processor.

    Returns:
        torch.Tensor: The image embeddings.
    """
    image = load_image(image_path)
    inputs = processor(images=[image], return_tensors="pt").to(model.device)

    with torch.no_grad():
        image_embeddings = model.get_image_features(**inputs)

    return image_embeddings

def batch_embed_images(image_paths, model, processor):
    """
    Generate embeddings for a batch of images.

    Args:
        image_paths (list): List of image file paths.
        model: The SigLIP2 model.
        processor: The SigLIP2 processor.

    Returns:
        torch.Tensor: Batch of image embeddings.
    """
    images = [load_image(image_path) for image_path in image_paths]
    inputs = processor(images=images, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        image_embeddings = model.get_image_features(**inputs)

    return image_embeddings

def save_embeddings(file_path, embeddings):
    """
    Save embeddings to a file.

    Args:
        file_path (str): Path to save the embeddings.
        embeddings (torch.Tensor): The embeddings to save.
    """
    np.save(file_path, embeddings.cpu().numpy())

def load_embeddings(file_path):
    """
    Load embeddings from a file.

    Args:
        file_path (str): Path to the embeddings file.

    Returns:
        torch.Tensor: The loaded embeddings.
    """
    return torch.tensor(np.load(file_path))