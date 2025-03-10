import os
import csv
import sys
import numpy as np
from tqdm import tqdm
from pathlib import Path
import multiprocessing as mp

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
# Suppress warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from core.metrics import Metrics
from core.models import StegoModel
from core import models_utils as ml_utils
from core.utils import load_image, save_image

# Global parameters
NUM_PROCESSES = 1
BATCH_SIZE = 8
OUTPUT_DIR = "/app/data/processed"
MODEL_NAME = "steguz"  # Configurable para futuros modelos
DATASET_NAME = "CFD"  # Configurable para futuros datasets
INPUT_FOLDER = f"{OUTPUT_DIR}/{DATASET_NAME}/{DATASET_NAME}_original"
OUTPUT_FOLDER = f"{OUTPUT_DIR}/{DATASET_NAME}/{DATASET_NAME}_{MODEL_NAME}/stego"
SECRET_IMAGE_PATH = "/app/data/secret/base/base_secret.jpg"
model_path_hide = Path(r'/app/models/hide/steguz/')
framework = 'tensorflow'
objective = 'hide'
dim_image = 256

# Inicializar función de pre-procesamiento para hiding
post_process_hide_func = lambda raw_stegos: ml_utils.post_process_hide_func_steguz(raw_stegos,
                                                                  scale=dim_image)
custom_objects = {"get_steguz_loss": ml_utils.get_steguz_loss}

def process_batch(image_paths, output_paths, model, csv_writer, progress_bar):
    """Procesa un batch de imágenes aplicando hiding."""
    covers = [load_image(img_path) for img_path in image_paths]
    
    secret_image = load_image(SECRET_IMAGE_PATH)
    stego_images = model.hide([c/255. for c in covers], 
                              secret_image/255., 
                              batch_size=BATCH_SIZE)
    
    for idx, (result, in_path, out_path) in enumerate(zip(stego_images, image_paths, output_paths)):
        stego = result["stego_images"][idx]
        cover = covers[idx]
        evaluator_hide = Metrics(cover.astype(np.uint8), stego.astype(np.uint8))
        metrics = evaluator_hide.compute_all()
        
        metadata = {"min_values": result["min_values"][idx], 
                    "max_values": result["max_values"][idx]}
        csv_writer.writerow([DATASET_NAME, MODEL_NAME, in_path, out_path, metadata, metrics])
        save_image(stego, out_path)
    
    progress_bar.update(len(image_paths))

def process_folders(input_folders, input_base, output_base, model, csv_writer, progress_bar):
    """Procesa imágenes manteniendo la estructura de carpetas."""
    os.makedirs(output_base, exist_ok=True)
    image_paths, output_paths = [], []
    
    for folder in input_folders:
        input_folder = os.path.join(input_base, folder)
        for root, _, files in os.walk(input_folder):
            relative_path = os.path.relpath(input_folder, input_base)
            output_subfolder = os.path.join(output_base, relative_path)
            os.makedirs(output_subfolder, exist_ok=True)
            
            for file in files:
                if file.endswith(".png"):  # Solo procesar imágenes PNG
                    input_path = os.path.join(root, file)
                    output_file = file.replace("original", "stego").replace("_224_224", "")
                    output_path = os.path.join(output_subfolder, output_file)
                    image_paths.append(input_path)
                    output_paths.append(output_path)
                    
                    if len(image_paths) == BATCH_SIZE:
                        process_batch(image_paths, output_paths, model, csv_writer, progress_bar)
                        image_paths, output_paths = [], []
    
    if image_paths:
        process_batch(image_paths, output_paths, model, progress_bar)

def main():
    """Ejecuta el proceso de hiding con multiprocesamiento."""
    csv_path = os.path.join(OUTPUT_DIR, DATASET_NAME, "hiding_summary.csv")
    file_exists = os.path.exists(csv_path)
    
    folders = next(os.walk(INPUT_FOLDER))[1]
    total_folders = len(folders)
    progress_bar = tqdm(total=total_folders, desc="Processing folders")
    
    chunk_size = total_folders // NUM_PROCESSES
    if chunk_size % 2 != 0:
        chunk_size += 1
    chunks = [folders[i:i + chunk_size] for i in range(0, total_folders, chunk_size)]
    
    models = [StegoModel(model_path_hide=model_path_hide,
                         framework=framework,
                         custom_objects=custom_objects,
                         objective=objective,
                         post_process_hide_func=post_process_hide_func) for _ in range(NUM_PROCESSES)]
    
    with open(csv_path, "a" if file_exists else "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists:
            csv_writer.writerow(["dataset", "model", "input_image", "output_image", "metadata", "metrics"])
        
        processes = []
        for i in range(NUM_PROCESSES):
            process = mp.Process(target=process_folders, 
                                 args=(chunks[i], INPUT_FOLDER, OUTPUT_FOLDER, 
                                       models[i], csv_writer, progress_bar))
            processes.append(process)
            process.start()
    
        for process in processes:
            process.join()
    
    progress_bar.close()
    print("✅ Hiding process completed.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Steganographic image hiding process")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()

    if args.debug:
        import debugpy
        # Start Debugging Server
        debugpy.listen(("0.0.0.0", 5678))
        print("✅ Waiting for debugger to attach...")
        debugpy.wait_for_client()
        print("Debugger attached! 🎯")

    main()
