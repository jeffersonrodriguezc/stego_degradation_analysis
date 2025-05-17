import os

def rename_images(folder):
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(".png"):
                old_path = os.path.join(root, file)
                
                # Eliminar el segmento "_original_224_224"
                new_name = file.replace("_original_224_224", "")
                
                # Mover "stego_" al final como "_stego"
                if new_name.startswith("stego_"):
                    new_name = new_name.replace("stego_", "", 1)  # Elimina "stego_" solo una vez
                    new_name = new_name.replace(".png", "_stego.png")  # Agrega "_stego" antes de la extensión
                
                new_path = os.path.join(root, new_name)
                
                # Renombrar el archivo si el nombre cambió
                if old_path != new_path:
                    os.rename(old_path, new_path)
                    print(f"Renombrado: {old_path} -> {new_path}")

# Uso del script
folder_path = "/app/data/processed/CFD_one_shot/CFD_one_shot_stegformer/stego"  # Cambia esto por la ruta real
rename_images(folder_path)