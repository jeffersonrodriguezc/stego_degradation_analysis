import os
import csv
import sys
import ast
import logging
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Suppress warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from core.metrics import Metrics
from core.models import StegoModel
from core import models_utils as ml_utils
from core.utils import load_image, save_image

# Global parameters
NUM_PROCESSES = 2
BATCH_SIZE = 8
OUTPUT_DIR = "/app/output/reveal"
MODEL_NAME = "steguz"  # Configurable for future models
DATASET_NAME = "CFD" # Configurable for future datasets
SECRET_IMAGE_PATH = "/app/data/secret/base/base_secret.jpg"
SECRET_IMAGE = load_image(SECRET_IMAGE_PATH)
input_type = "transformations" # if we are gonna use the stego images or the transformations
transformation = None #"resize" ...
summary_name = "transformations_summary.csv" # hiding for stego, transformations for transformations
SUMMARY_FILE_PATH = f"/app/data/processed/{DATASET_NAME}/{summary_name}"
model_path_reveal = Path(r'/app/models/reveal/steguz/')
framework = 'tensorflow'
objective = 'reveal'
dim_image = 224

# Initialize preprocessing function for revealing
pre_process_reveal_func = lambda stegos_dict: ml_utils.pre_process_reveal_func_steguz(stegos_dict,
                                                                                 scale=dim_image)
custom_objects = {"get_steguz_loss": ml_utils.get_steguz_loss}

def process_batch(image_paths, output_paths, model, csv_writer, summary, progress_bar):
    """Processes a batch of images using the revealing model."""
    # Load images 
    images = []
    min_values = []
    max_values = []
    for img_path in image_paths:
        image = load_image(img_path)
        row = summary[summary['output_image'] == img_path]
        metadata = ast.literal_eval(row.iloc[0]['metadata'])
        min_value, max_value = metadata["min_values"], metadata["max_values"]
        min_values.append(min_value)
        max_values.append(max_value)
        images.append(image)

    recovered_secrets = model.reveal({"stego_images":images,
                                      "min_values":min_values,
                                       "max_values":max_values}, batch_size=8)
    
    # Save recovered images
    for recovered_secret, in_path, out_path in zip(recovered_secrets['secret_images'], image_paths, output_paths):
        evaluator_reveal = Metrics(SECRET_IMAGE, (recovered_secret*255).astype(np.uint8))
        metrics = evaluator_reveal.compute_all()
        csv_writer.writerow([DATASET_NAME, MODEL_NAME, in_path, out_path, metrics])
        save_image((recovered_secret*255.).astype(np.uint8), out_path)
    
    # Update progress bar
    progress_bar.update(len(image_paths))
    
def process_folders(input_folders, input_base, output_base, 
                    model, csv_writer, hiding_summary, progress_bar):
    """Processes all images within a folder while maintaining the structure."""
    os.makedirs(output_base, exist_ok=True)
    image_paths, output_paths = [], []
    
    for folder in input_folders:
        input_folder = os.path.join(input_base, folder)
        for root, _, files in os.walk(input_folder):
            relative_path = os.path.relpath(root, input_base)
            output_subfolder = os.path.join(output_base, relative_path)
            os.makedirs(output_subfolder, exist_ok=True)
            for file in files:
                if file.endswith(".png"):  # Only process PNG images
                    input_path = os.path.join(root, file)
                    output_file = file.replace("stego", "recovered") \
                        if "stego" in file else f"{file.split('.')[0]}_recovered.png"
                    output_path = os.path.join(output_subfolder, output_file)
                    image_paths.append(input_path)
                    output_paths.append(output_path)
                
                    # Process in batches of BATCH_SIZE
                    if len(image_paths) == BATCH_SIZE:
                        process_batch(image_paths, output_paths, model, csv_writer, hiding_summary, progress_bar)
                        image_paths, output_paths = [], []
    
    # Process remaining images
    if image_paths:
        process_batch(image_paths, output_paths, model, csv_writer, hiding_summary, progress_bar)
    progress_bar.close()

def main(input_type, transformation=None, debug_mode=False):
    """Executes the revealing process on the `stego` or `transformations` folder with parallelization."""
    input_base = "/app/data/processed/{}/{}_{}/{}".format(DATASET_NAME, DATASET_NAME,
                                                     MODEL_NAME, input_type)
    output_base = os.path.join(OUTPUT_DIR, "{}/{}_{}/{}".format(DATASET_NAME, DATASET_NAME,
                                                               MODEL_NAME, input_type))
    csv_path = os.path.join(OUTPUT_DIR, DATASET_NAME, "revealing_summary.csv")
    summary = pd.read_csv(SUMMARY_FILE_PATH)
    
    # Determine input and output folders
    if input_type == "stego":
        input_folder = input_base
        output_folder = output_base
    elif input_type == "transformations":
        if transformation:
            input_folder = os.path.join(input_base, transformation)
            output_folder = os.path.join(output_base, transformation)
        else:
            input_folder = input_base
            output_folder = output_base
    else:
        raise ValueError("Invalid input type. Must be 'stego' or 'transformations'.")
    
    os.makedirs(output_base, exist_ok=True)
    # Check if the CSV file already exists
    file_exists = os.path.exists(csv_path)
    
    # Count folder for progress bar
    # if it is stego folder, the folders are the persons
    # if it is transformations folder, the folders are the transformations
    # if it is a specific transformation, the folders are the variations
    folders = next(os.walk(input_folder))[1]
    total_folders = len(folders)
    if transformation:
        total_images = summary[summary["transformation"]==transformation].shape[0]
    else:
        total_images = summary.shape[0]

    progress_bar = tqdm(total=total_images, desc="Processing images")
    
    # Initialize multiple revealing models
    model = StegoModel(model_path_reveal=model_path_reveal,
                        framework=framework,
                        custom_objects=custom_objects,
                        objective=objective,
                        pre_process_reveal_func=pre_process_reveal_func)
    
    with open(csv_path, "a" if file_exists else "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists:
            csv_writer.writerow(["dataset", "model", "input_image", "output_image", "metrics"])
        
        process_folders(folders, input_folder, output_folder, 
                        model, csv_writer, summary, progress_bar)

    print("✅ Revealing process completed.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Steganographic image revealing process")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (single process)")
    args = parser.parse_args()

    if args.debug:
        import debugpy
        # Start Debugging Server
        debugpy.listen(("0.0.0.0", 5678))
        print("✅ Waiting for debugger to attach...")
        debugpy.wait_for_client()
        print("Debugger attached! 🎯")

    main(input_type, transformation, args.debug)