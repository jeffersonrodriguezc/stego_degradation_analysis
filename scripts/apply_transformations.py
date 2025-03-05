import os
import multiprocessing
import cv2
import numpy as np
from core.transformations import apply_transformation_pipeline  # Importa la función desde el módulo de transformaciones

# Definir rutas
DATA_DIR = "data/processed"
OUTPUT_DIR = "data/transformed"

# Asegurar que la carpeta de salida exista
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_image(person_folder: str):
    """
    Procesa todas las imágenes dentro de la carpeta de una persona.
    Aplica transformaciones y guarda las imágenes en la carpeta de salida.
    """
    person_path = os.path.join(DATA_DIR, person_folder)
    output_person_path = os.path.join(OUTPUT_DIR, person_folder)

    # Asegurar que la carpeta de salida de la persona exista
    os.makedirs(output_person_path, exist_ok=True)

    # Obtener la lista de imágenes Stego en la carpeta de la persona
    for filename in os.listdir(person_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(person_path, filename)
            output_path = os.path.join(output_person_path, filename)

            # Cargar imagen
            image = cv2.imread(image_path)
            if image is None:
                print(f"[ERROR] No se pudo cargar la imagen: {image_path}")
                continue

            # Aplicar transformaciones
            transformed_image = apply_transformation_pipeline(image)

            # Guardar imagen transformada
            cv2.imwrite(output_path, transformed_image)
            print(f"[INFO] Imagen procesada y guardada en: {output_path}")

def main():
    """
    Encuentra todas las carpetas en data/processed y las procesa en paralelo.
    """
    person_folders = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))]

    print(f"[INFO] Se encontraron {len(person_folders)} carpetas de personas para procesar.")

    # Usar multiprocessing para acelerar el procesamiento
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(process_image, person_folders)

if __name__ == "__main__":
    main()
