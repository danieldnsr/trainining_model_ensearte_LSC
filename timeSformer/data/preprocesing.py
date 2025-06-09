import av 
import av.error
import numpy as np
from PIL import Image
import torch
from typing import List, Optional, Tuple
import torchvision.transforms as T
import torchvision.transforms.functional as F

def read_and_sample_video_pyav(video_path: str, num_frames: int) -> Optional[List[np.ndarray]]:
    """
    Reads a video file using PyAV and samples a fixed number of frames.

    Args:
        video_path (str): Path to the video file.
        num_frames (int): Desired number of frames to sample.

    Returns:
        Optional[List[np.ndarray]]: A list of sampled RGB frames as NumPy arrays,
        or None if an error occurs or the video cannot be decoded.
    """
    try:
        container = av.open(video_path)  # Abre el video usando PyAV
        
        # Decodifica todos los frames como arrays RGB
        all_frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
        container.close()

        total_frames = len(all_frames)
        if total_frames == 0:
            print(f"Error: No se decodificaron frames de {video_path}")
            return None

        if total_frames >= num_frames:
            # Muestra num_frames uniformemente distribuidos
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            sampled_frames = [all_frames[i] for i in indices]
        else:
            # Si hay menos frames que los deseados, repite el último
            sampled_frames = all_frames
            while len(sampled_frames) < num_frames:
                sampled_frames.append(all_frames[-1])
        
        return sampled_frames

    except av.error as e:
        print(f"Error de PyAV al procesar {video_path}: {e}")
        return None
    except Exception as e:
        print(f"Error inesperado al procesar {video_path} con PyAV: {e}")
        return None


def apply_train_augmentations(
    frames: List[np.ndarray], 
    output_size: Tuple[int, int]
) -> List[np.ndarray]:
    """
    Applies spatial augmentations consistently across all frames in a video sequence.

    Args:
        frames (List[np.ndarray]): Input frames as NumPy arrays.
        output_size (Tuple[int, int]): Desired output size (H, W).

    Returns:
        List[np.ndarray]: Augmented frames.
    """
    pil_frames = [Image.fromarray(f) for f in frames]

    # crop
    crop_params = T.RandomResizedCrop.get_params(
        pil_frames[0], scale=(0.8, 1.0), ratio=(0.9, 1.1)
    )
    pil_frames = [F.resized_crop(img, *crop_params, output_size, antialias=True) for img in pil_frames]

    # color jitter
    color_jitter = T.ColorJitter(
        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02
    )
    pil_frames = [color_jitter(img) for img in pil_frames]

    return [np.array(f) for f in pil_frames]


def process_with_hf(frames: List, image_processor) -> torch.Tensor:
    """
    Applies the HuggingFace image processor to a sequence of frames.

    Args:
        frames (List): A list of PIL or NumPy frames.
        image_processor: HuggingFace processor to apply (e.g., ViT or VideoMAE processor).

    Returns:
        torch.Tensor: The processed tensor ready for model input.
    """
    inputs = image_processor(frames, return_tensors="pt")  # Procesa los frames en batch
    return inputs['pixel_values'].squeeze(0)  # Elimina el batch extra (1, T, C, H, W) -> (T, C, H, W)


def create_map_function(image_processor, num_frames: int, resize_to: Tuple[int, int], is_train: bool):
    """
    Creates a callable function to be passed to `datasets.Dataset.map()`.
    This function reads and processes a video sample.

    Args:
        image_processor: HuggingFace image processor.
        num_frames (int): Number of frames to sample from each video.
        resize_to (Tuple[int, int]): Target frame size (e.g., (224, 224)).
        is_train (bool): Whether to apply training-time augmentations.

    Returns:
        Callable: A function that takes a dataset example and returns processed input.
    """
    def map_fn(example):
        # Lee y muestra los frames del video
        frames = read_and_sample_video_pyav(example['video_path'], num_frames)
        
        if frames is None:
            print(f"Error o video corto, saltando: {example['video_path']}")
            return {"pixel_values": None}

        # Aplica aumentaciones si es para entrenamiento
        if is_train:
            frames = apply_train_augmentations(frames, resize_to)

        # Procesamiento final con HuggingFace image processor
        pixel_values = process_with_hf(frames, image_processor)

        return {"pixel_values": pixel_values}
        
    return map_fn
