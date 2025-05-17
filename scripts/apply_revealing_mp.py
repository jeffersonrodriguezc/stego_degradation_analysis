import os
import csv
import sys
from tqdm import tqdm
from pathlib import Path
import multiprocessing as mp

import warnings
# Suppress warnings
warnings.filterwarnings("ignore")


sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from core.models import StegoModel
from core import models_utils as ml_utils
from core.utils import load_image, save_image

# Global parameters
NUM_PROCESSES = 2
BATCH_SIZE = 8
OUTPUT_DIR = "/app/output/reveal"
MODEL_NAME = "steguz"  # Configurable for future models
DATASET_NAME = "CFD" # Configurable for future datasets
input_type = "stego" # if we are gonna use the stego images or the transformations
transformation = None
model_path_reveal = Path(r'/app/models/hide/steguz/')
framework = 'tensorflow'
objective = 'reveal'
dim_image = 256
pre_process_reveal_func = lambda stegos_dict: ml_utils.pre_process_reveal_func_steguz(stegos_dict,
                                                                                 scale=dim_image)
custom_objects = {"get_steguz_loss": ml_utils.get_steguz_loss}

def process_batch(image_paths, output_paths, model, progress_bar):
    """Processes a batch of images using the revealing model."""
    # Load images 
    images = []
    min_values = []
    max_values = []
    for img_path in image_paths:
        image, metadata = load_image(img_path, operation="reveal")
        min_value, max_value = metadata["normalization_values"].split(',')
        min_values.append(min_value)
        max_values.append(max_value)
        images.append(image)

    recovered_secrets = model.reveal({"stego_images":images,
                                      "min_values":min_values,
                                       "max_values":max_values}, batch_size=8)
    
    # Save recovered images
    for img, out_path in zip(recovered_secrets, output_paths):
        save_image(img, out_path)
    
    # Update progress bar
    progress_bar.update(len(image_paths))

def process_folders(input_folders, input_base, output_base, 
                    model, csv_writer, DATASET_NAME, MODEL_NAME, progress_bar):
    """Processes all images within a folder while maintaining the structure."""
    os.makedirs(output_base, exist_ok=True)
    image_paths, output_paths = [], []
    
    for folder in input_folders:
        input_folder = os.path.join(input_base, folder)
        for root, _, files in os.walk(input_folder):
            relative_path = os.path.relpath(input_folder, input_base)
            output_subfolder = os.path.join(output_base, relative_path)
            #os.makedirs(output_subfolder, exist_ok=True)
            for file in files:
                if file.endswith(".png"):  # Only process PNG images
                    input_path = os.path.join(root, file)
                    output_file = file.replace("stego", "recovered") \
                        if "stego" in file else f"{file.split('.')[0]}_recovered.png"
                    output_path = os.path.join(output_subfolder, output_file)
                    image_paths.append(input_path)
                    output_paths.append(output_path)
                    csv_writer.writerow([DATASET_NAME, MODEL_NAME, input_path, output_path])
                
                    # Process in batches of BATCH_SIZE
                    if len(image_paths) == BATCH_SIZE:
                        process_batch(image_paths, output_paths, model, progress_bar)
                        image_paths, output_paths = [], []
    
    # Process remaining images
    if image_paths:
        pass
        #process_batch(image_paths, output_paths, model, progress_bar)

def main(input_type, transformation=None, debug_mode=False):
    """Executes the revealing process on the `stego` or `transformations` folder with parallelization."""
    input_base = "/app/data/processed/{}/{}_{}/{}".format(DATASET_NAME, DATASET_NAME,
                                                     MODEL_NAME, input_type)
    output_base = os.path.join(OUTPUT_DIR, "{}/{}_{}/{}".format(DATASET_NAME, DATASET_NAME,
                                                               MODEL_NAME, input_type))
    csv_path = os.path.join(OUTPUT_DIR,"revealing.csv")
    
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
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Check if the CSV file already exists
    file_exists = os.path.exists(csv_path)
    
    # Count folder for progress bar
    # if it is stego folder, the folders are the persons
    # if it is transformations folder, the folders are the transformations
    # if it is a specific transformation, the folders are the variations
    folders = next(os.walk(input_folder))[1]
    total_folders = len(folders)
    progress_bar = tqdm(total=total_folders, desc="Processing folders")

    # create the chunks of the folders for the parallel processing
    chunk_size = total_folders // NUM_PROCESSES
    # if the chunk size is odd, we add 1 to make it even
    if chunk_size % 2 != 0:
        chunk_size += 1
    chunks = [folders[i:i + chunk_size] for i in range(0, total_folders, chunk_size)]
    
    # Initialize multiple revealing models
    models = [StegoModel(model_path_reveal=model_path_reveal,
                        framework=framework,
                        custom_objects=custom_objects,
                        objective=objective,
                        pre_process_reveal_func=pre_process_reveal_func) for _ in range(NUM_PROCESSES)]
    
    if debug_mode:
        print("🔍 Debug Mode Active: Running in Single Process")
        print("📂 Input Folder:", input_folder)
        print("📂 Output Folder:", output_folder)
        print("📄 File exist:", file_exists)
        print("📄 CSV Path:", csv_path)
        print("🔢 Total Folders:", total_folders)
        print("🔢 Chunk Size:", chunk_size)
        print("🔢 folders (sample):", folders[:10])
        for chunk in chunks:
            print("📁 Chunk:", len(chunk))
        for idx, model in enumerate(models):
            print(f"🧠 Model {idx} GPU memory usage {model.gpu_memory_usage()}")
    
    with open(csv_path, "a" if file_exists else "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists:
            csv_writer.writerow(["dataset", "model", "input_image", "output_image"])
        
        # Start parallel processing
        processes = []
        for i in range(NUM_PROCESSES):
            process = mp.Process(target=process_folders, 
                                args=(chunks[i], input_folder, output_folder, 
                                      models[i], csv_writer, DATASET_NAME, MODEL_NAME, progress_bar))
            processes.append(process)
            process.start()
    
        for process in processes:
            process.join()
    progress_bar.close()
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