
import os
import cv2
import math
import yaml
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from PIL.PngImagePlugin import PngInfo

def load_yaml_config(config_path, key):
    if not os.path.exists(config_path):  # Check if the file exists
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    return config[key]

def load_image(image_path):
    """
    Reads an image from a file path and returns it as a numpy array.
    :param image_path: Image path.
    :return: np.array Image readed.
    """
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

def load_image_metadata(image_path, operation=None, metadata=None):
    """
    Reads an image from a file path and returns it as a numpy array.    
    :param image_path: Image path.
    :return: np.array Image readed.
    """
    image = Image.open(image_path)
    if operation == "reveal":
        metadata = image.text  # Diccionario con los metadatos

    return np.array(image), metadata

def save_image(image, output_path):
    """
    Saves an image using OpenCV.
    :param image: Image to save.
    :param output_path: Path where the image will be saved.
    """
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def save_image_metadata(image, output_path, metadata={}):
    """
    Saves an image in PNG format with metadata.

    :param image: Image in PIL format (Image.fromarray()).
    :param output_path: Path where the image will be saved.
    :param metadata: Dictionary containing the metadata.
    """
    try:
        # Create a PngInfo object to store metadata
        png_metadata = PngInfo()

        # Add metadata to the PngInfo object
        for key, value in metadata.items():
            png_metadata.add_text(key, str(value))  # Convert values to string

        # Save the image with metadata
        image.save(output_path, "PNG", pnginfo=png_metadata)
        print(f"Image saved at: {output_path} with metadata: {metadata}")

    except Exception as e:
        print(f"Error saving the image: {e}")

def apply_transformations_and_display(image_path, transformer, pipeline):
    """
    Applies a series of transformations to a stego image and displays the original image along with
    all transformed variations in a Jupyter Notebook using matplotlib.

    Parameters:
    -----------
    image_path : str
        Path to the stego image.
    transformer : object
        An object that implements the 'apply_pipeline' method to apply transformations.
    pipeline : list of dict
        A list of dictionaries defining each transformation operation. Each dictionary should include:
            - 'name': The transformation name.
            - 'variations': A list of parameter sets.
            - 'titles': A list of corresponding titles for each variation.

    Returns:
    --------
    dict
        A dictionary mapping each transformation title to its resulting transformed image(s).
    """
    # Read the original image using OpenCV
    image = cv2.imread(image_path)

    # Apply the pipeline transformations
    results = transformer.apply_pipeline(image, pipeline)

    # Prepare the list of images to display
    display_images = [("Original", image)]
    
    for step in pipeline:
        name = step['name']
        variations = results.get(name, [])
        titles = step.get('titles', [])

        for img, title in zip(variations, titles):
            display_images.append((title, img))
    
    # Determine grid layout dynamically: Arrange in a grid of up to 3 columns
    total_images = len(display_images)
    cols = min(4, total_images)
    rows = math.ceil(total_images / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).flatten()

    # Display images with titles
    for ax, (title, img) in zip(axes, display_images):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(title)
        ax.axis("off")

    # Hide any unused subplots
    for ax in axes[len(display_images):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    return results