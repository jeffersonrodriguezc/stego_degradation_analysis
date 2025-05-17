import sys
import os
import pandas as pd
import random
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import cv2
import ast

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from core.metrics import Metrics
from core.utils import load_image

# Rutas base
DATASET_NAME = 'CFD_one_shot'
STEGO_MODEL = 'steguz'
MANIPULATION_NAME = 'morphing'
MANIPULATION_MODEL = 'morphGEN'
stego_base_dir = f"/app/data/processed/{DATASET_NAME}/{DATASET_NAME}_{STEGO_MODEL}/stego"
original_base_dir = f"/app/data/processed/{DATASET_NAME}/{DATASET_NAME}_original"
output_base_dir = f"/app/data/processed/{DATASET_NAME}/{DATASET_NAME}_{STEGO_MODEL}/manipulations/{MANIPULATION_MODEL}"
# Cargar el CSV con la información de train/test
data_csv = f"/app/data/processed/{DATASET_NAME}/data_splits.csv"
df = pd.read_csv(data_csv)
morphgen_script = "/app/morphgen.py"
alpha = 0.9  # Alpha value
csv_output = f"/app/data/processed/{DATASET_NAME}/manipulations_summary.csv"
parameters = {'alpha':alpha}

# Crear la carpeta de salida
os.makedirs(output_base_dir, exist_ok=True)

# Diccionario para agrupar subjects por su set (train o test)
subjects_by_set = {}
for _, row in df.iterrows():
    subject = row["subject"]
    set_type = row["set"]
    if set_type not in subjects_by_set:
        subjects_by_set[set_type] = []
    subjects_by_set[set_type].append(subject)


# Cargar CSV existente si ya está creado
try:
    existing_df = pd.read_csv(csv_output)
except FileNotFoundError:
    existing_df = pd.DataFrame(columns=["dataset", "input_image", "output_image", "manipulation", 
                                        "variant", "parameters", "model_manipulation", "model", "metrics", "metadata", "status"])

csv_hiding_summary = pd.read_csv(f"/app/data/processed/{DATASET_NAME}/hiding_summary.csv")
csv_records = []

def compute_metrics(img1_path, img2_path):
    """Calcula métricas entre dos imágenes."""
    img1 = load_image(img1_path)
    img2 = load_image(img2_path)
    if img1 is None or img2 is None:
        return "Error: Image not found"
    evaluator = Metrics(img1, img2)
    return evaluator.compute_all()

def generate_morphs(stego_subject, original_subject, stego_image, 
                    original_image, output_dir, morph_variant, metadata):
    """Genera morphing entre imágenes stego y originales y almacena la información en CSV."""
    os.makedirs(output_dir, exist_ok=True)
    
    command = ["python3", morphgen_script,
               "--from_images", original_image, stego_image,
               "--output_dir", output_dir,
               "--alpha", str(alpha)]
    
    print(f"Ejecutando: {' '.join(command)}")
    subprocess.run(command)
    
    # Identificar la imagen de salida final (no _stego.png)
    output_images = [f for f in os.listdir(output_dir) if f.endswith(".png") and not f.endswith("_stego.png")]
    
    if not output_images:
        status = "Error: No output image found"
        csv_records.append([DATASET_NAME, stego_image, "N/A", MANIPULATION_NAME, 
                            morph_variant, parameters, MANIPULATION_MODEL, STEGO_MODEL, "N/A", metadata, status])
        return
    
    final_output_image = os.path.join(output_dir, output_images[0])
    metrics = compute_metrics(stego_image, final_output_image)

        # Eliminar la imagen generada con _stego.png
    for file in os.listdir(output_dir):
        if file.endswith("_stego.png"):
            os.remove(os.path.join(output_dir, file))
    
    csv_records.append([DATASET_NAME, stego_image, final_output_image, MANIPULATION_NAME, morph_variant, parameters, 
                        MANIPULATION_MODEL, STEGO_MODEL, metrics, metadata, "Success"])

def process_subject(subject):
    """Procesa un sujeto y genera morphs con otros sujetos dentro del mismo set."""
    stego_subject_path = os.path.join(stego_base_dir, subject)
    original_subject_path = os.path.join(original_base_dir, subject)

    if not os.path.isdir(stego_subject_path) or not os.path.isdir(original_subject_path):
        return  # Saltar si no es una carpeta válida
    
    # Obtener la lista de imágenes stego dentro de la carpeta
    stego_images = [f for f in os.listdir(stego_subject_path) if f.endswith(".png")]
    
    for stego_image in stego_images:
        stego_image_path = os.path.join(stego_subject_path, stego_image)
        row = csv_hiding_summary[csv_hiding_summary['output_image'] == stego_image_path]
        metadata = ast.literal_eval(row.iloc[0]['metadata'])
        
        # Obtener la categoría train/test del subject actual
        subject_set = df[df["subject"] == subject]["set"].values[0]
        
        # Seleccionar 8 sujetos diferentes dentro del mismo set, excluyendo el mismo sujeto
        available_subjects = [s for s in subjects_by_set[subject_set] if s != subject]
        selected_subjects = random.sample(available_subjects, min(8, len(available_subjects)))
        
        # Crear carpeta de salida para este sujeto
        subject_output_dir = os.path.join(output_base_dir, subject)
        os.makedirs(subject_output_dir, exist_ok=True)
        
        for idx, other_subject in enumerate(selected_subjects, start=1):
            other_original_path = os.path.join(original_base_dir, other_subject)
            
            # Verificar si el sujeto seleccionado tiene una imagen original
            other_original_images = [f for f in os.listdir(other_original_path) if f.endswith(".png")]
            if not other_original_images:
                continue
            
            original_image_path = os.path.join(other_original_path, other_original_images[0])
            
            morph_output_dir = os.path.join(subject_output_dir, f"morph_{idx}")
            generate_morphs(subject, other_subject, stego_image_path, 
                            original_image_path, morph_output_dir, f"morph_{idx}", metadata)

def main():
    num_workers = max(1, int(multiprocessing.cpu_count() * 0.6))
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        executor.map(process_subject, os.listdir(stego_base_dir))
    
    # Guardar CSV sin sobrescribir, agregando los nuevos resultados
    new_df = pd.DataFrame(csv_records, columns=existing_df.columns)
    updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    updated_df.to_csv(csv_output, index=False)
    print(f"CSV actualizado en: {csv_output}")
    print("Proceso de morphing completado.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Steganographic image manipulation process")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()

    if args.debug:
        import debugpy
        debugpy.listen(("0.0.0.0", 5678))
        print("✅ Waiting for debugger to attach...")
        debugpy.wait_for_client()
        print("Debugger attached! 🎯")

    main()
