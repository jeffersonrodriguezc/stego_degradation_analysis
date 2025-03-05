import os
import multiprocessing
import cv2
from PIL import Image
import numpy as np

# Definir rutas
RAW_DIR = "data/raw/CFD_Dataset/CFD_Version_3_0/images/CFD"
PROCESSED_DIR = "data/processed/CFD/CFD_original"

# Asegurar que la carpeta de salida exista
os.makedirs(PROCESSED_DIR, exist_ok=True)

def resize_and_save(person_folder: str):
    """
    Toma las imágenes en `data/raw/{persona}/`, las redimensiona a 224x224 píxeles
    y las guarda en `data/processed/{persona}/` con el nombre `original_224_224.png`.
    """
    raw_person_path = os.path.join(RAW_DIR, person_folder)
    processed_person_path = os.path.join(PROCESSED_DIR, person_folder)

    # Asegurar que la carpeta de salida de la persona exista
    os.makedirs(processed_person_path, exist_ok=True)

    # Buscar imágenes en la carpeta de la persona
    for filename in os.listdir(raw_person_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(raw_person_path, filename)
            output_path = os.path.join(processed_person_path, filename[:-4]+"_original_224_224.png")

            # Cargar imagen
            image = cv2.imread(image_path)
            #image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            #image = Image.open(image_path)

            if image is None:
                print(f"[ERROR] No se pudo cargar la imagen: {image_path}")
                continue

            # Redimensionar imagen a 224x224
            #resized_image = image.resize(size=(224, 224))
            #resized_image = cv2.cvtColor(np.array(resized_image), cv2.COLOR_RGB2BGR)
            resized_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

            # Guardar imagen redimensionada
            cv2.imwrite(output_path, resized_image)
            #resized_image.save(output_path)
            print(f"[INFO] Imagen redimensionada y guardada en: {output_path}")

            # Solo guardamos una imagen por carpeta
            #break 

def main():
    """
    Encuentra todas las carpetas en `data/raw/` y procesa las imágenes en paralelo.
    """
    person_folders = [f for f in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, f))]

    print(f"[INFO] Se encontraron {len(person_folders)} carpetas de personas para procesar.")

    # Usar multiprocessing para acelerar el procesamiento
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()-2) as pool:
        pool.map(resize_and_save, person_folders)

if __name__ == "__main__":
    main()
