import os
import random
from typing import List, Tuple

from PIL import Image


def generate_synthetic_images(
    output_dir: str = "outputs/synthetic_images",
    num_images: int = 10,
    image_size: Tuple[int, int] = (64, 64),
    seed: int = 42,
) -> List[str]:
    """
    Minimal GAN wiring placeholder.
    Generates simple noise images to validate the pipeline end-to-end.
    Replace with a real GAN sampler in the full training integration.
    """
    os.makedirs(output_dir, exist_ok=True)

    rng = random.Random(seed)
    saved_paths: List[str] = []

    for i in range(num_images):
        pixels = [
            (rng.randrange(256), rng.randrange(256), rng.randrange(256))
            for _ in range(image_size[0] * image_size[1])
        ]
        img = Image.new("RGB", image_size)
        img.putdata(pixels)

        file_name = f"gan_synth_{i:03d}.png"
        file_path = os.path.join(output_dir, file_name)
        img.save(file_path)
        saved_paths.append(file_path)

    return saved_paths
