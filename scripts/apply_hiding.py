import os
import csv
import sys
import logging
import warnings
import numpy as np
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
BATCH_SIZE = 8
OUTPUT_DIR = "/app/data/processed"
MODEL_NAME = "steguz"  # Configurable for future models
DATASET_NAME = "CFD"  # Configurable for future datasets
INPUT_FOLDER = f"{OUTPUT_DIR}/{DATASET_NAME}/{DATASET_NAME}_original"
OUTPUT_FOLDER = f"{OUTPUT_DIR}/{DATASET_NAME}/{DATASET_NAME}_{MODEL_NAME}/stego"
SECRET_IMAGE_PATH = "/app/data/secret/base/base_secret.jpg"
SECRET_IMAGE = load_image(SECRET_IMAGE_PATH)
model_path_hide = Path(r'/app/models/hide/steguz/')
framework = 'tensorflow'
objective = 'hide'
dim_image = 224

# Initialize preprocessing function for hiding
post_process_hide_func = lambda raw_stegos: ml_utils.post_process_hide_func_steguz(raw_stegos,
                                                                  scale=dim_image)
custom_objects = {"get_steguz_loss": ml_utils.get_steguz_loss}

def process_batch(image_paths, output_paths, model, csv_writer, progress_bar):
    """Procesa un batch de imágenes aplicando hiding."""
    covers = [load_image(img_path) for img_path in image_paths]
    
    stego_results = model.hide([c.astype('float32') / 255.0 for c in covers], 
                              SECRET_IMAGE.copy().astype('float32') / 255.0, 
                              batch_size=BATCH_SIZE)

    for idx, (stego, in_path, out_path) in enumerate(zip(stego_results["stego_images"], 
                                                          image_paths, 
                                                          output_paths)):
        cover = covers[idx]
        evaluator_hide = Metrics(cover.astype(np.uint8), stego.astype(np.uint8))
        metrics = evaluator_hide.compute_all()
        
        metadata = {"min_values": stego_results["min_values"][idx], 
                    "max_values": stego_results["max_values"][idx]}
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
        process_batch(image_paths, output_paths, model, csv_writer, progress_bar)

    progress_bar.close()

def main():
    """Ejecuta el proceso de hiding con multiprocesamiento."""
    csv_path = os.path.join(OUTPUT_DIR, DATASET_NAME, "hiding_summary.csv")
    file_exists = os.path.exists(csv_path)
    
    folders = next(os.walk(INPUT_FOLDER))[1]
    total_folders = len(folders)
    
    total_images = 0
    for root, _, files in os.walk(INPUT_FOLDER):
        total_images += sum(1 for file in files if file.lower().endswith('.png'))

    progress_bar = tqdm(total=total_images, desc="Processing images")
    
    model = StegoModel(model_path_hide=model_path_hide,
                         framework=framework,
                         custom_objects=custom_objects,
                         objective=objective,
                         post_process_hide_func=post_process_hide_func)
    
    with open(csv_path, "a" if file_exists else "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists:
            csv_writer.writerow(["dataset", "model", "input_image", "output_image", "metadata", "metrics"])
        
        process_folders(folders, INPUT_FOLDER, OUTPUT_FOLDER, 
                            model, csv_writer, progress_bar)
    
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
