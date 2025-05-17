import sys
import os
import pandas as pd
import random
import subprocess
from pathlib import Path

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

# Cargar el CSV con la información de train/test
data_csv = "/app/data/processed/CFD_one_shot/data_splits.csv"
df = pd.read_csv(data_csv)

# Diccionario para agrupar subjects por su set (train o test)
subjects_by_set = {}
for _, row in df.iterrows():
    subject = row["subject"]
    set_type = row["set"]
    if set_type not in subjects_by_set:
        subjects_by_set[set_type] = []
    subjects_by_set[set_type].append(subject)

# Rutas base
stego_base_dir = "/app/data/processed/CFD_one_shot/CFD_one_shot_stegformer/stego"
original_base_dir = "/app/data/processed/CFD_one_shot/CFD_one_shot_original"
output_base_dir = "/app/data/processed/CFD_one_shot/CFD_one_shot_stegformer/manipulations/morphGEN"
morphgen_script = "/app/morphgen.py"
alpha = 0.5  # Alpha value

# Crear la carpeta de salida
os.makedirs(output_base_dir, exist_ok=True)

def generate_morphs(stego_subject, original_subject, stego_image, original_image, output_dir):
    """Genera morphing entre imágenes stego y originales."""
    os.makedirs(output_dir, exist_ok=True)
    
    command = ["python3", morphgen_script,
               "--from_images", original_image, stego_image,
               "--output_dir", output_dir,
               "--alpha", str(alpha)]
    
    print(f"Eseguendo: {' '.join(command)}")
    subprocess.run(command)

    # Eliminar la imagen generada con _stego.png
    for file in os.listdir(output_dir):
        if file.endswith("_stego.png"):
            os.remove(os.path.join(output_dir, file))


def main ():
    # Iterar sobre todas las carpetas de imágenes stego
    for subject in os.listdir(stego_base_dir):
        stego_subject_path = os.path.join(stego_base_dir, subject)
        original_subject_path = os.path.join(original_base_dir, subject)
        
        if not os.path.isdir(stego_subject_path) or not os.path.isdir(original_subject_path):
            continue  # Saltar si no es una carpeta válida
        
        # Obtener la lista de imágenes stego dentro de la carpeta
        stego_images = [f for f in os.listdir(stego_subject_path) if f.endswith(".png")]
        
        for stego_image in stego_images:
            stego_image_path = os.path.join(stego_subject_path, stego_image)
            
            # Obtener la categoría train/test del subject actual
            subject_set = df[df["subject"] == subject]["set"].values[0]
            
            # Seleccionar 8 sujetos diferentes dentro del mismo set
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
                generate_morphs(subject, other_subject, stego_image_path, original_image_path, morph_output_dir)

    print("Proceso de morphing completado.")

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