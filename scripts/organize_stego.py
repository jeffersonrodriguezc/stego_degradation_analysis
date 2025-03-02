from pathlib import Path
import shutil

def separar_stego(dataset_path: str, dataset_stego_path: str):
    # Convertir a objetos Path y obtener rutas absolutas
    dataset = Path(dataset_path).resolve()
    dataset_stego = Path(dataset_stego_path).resolve()

    # Crear la carpeta destino para las imágenes stego si no existe
    dataset_stego.mkdir(parents=True, exist_ok=True)

    # Iterar sobre cada subcarpeta del dataset original (cada personaje)
    for persona_folder in dataset.iterdir():
        if not persona_folder.is_dir():
            continue  # Se ignoran archivos que estén en la raíz

        # Crear la carpeta correspondiente en el dataset stego
        stego_persona_folder = dataset_stego / persona_folder.name
        stego_persona_folder.mkdir(parents=True, exist_ok=True)

        # Recorrer cada archivo en la carpeta del personaje
        for file_path in persona_folder.iterdir():
            if file_path.is_file():
                # Si el nombre (sin extensión) contiene "_stego", es una imagen stego
                if "_stego" in file_path.stem:
                    destino = stego_persona_folder / file_path.name
                    print(f"Moviendo {file_path} a {destino}")
                    # Mueve el archivo: lo elimina del dataset original y lo copia a la estructura stego
                    shutil.move(str(file_path), str(destino))
                # Las imágenes que no son stego se dejan en el dataset original

if __name__ == "__main__":
    # Define la ruta del dataset original (por ejemplo, "CFD") y el destino para las imágenes stego (por ejemplo, "CFD_stego")
    dataset_original = r"..\data\raw\CFD_Dataset\CFD_Version_3_0\Images\CFD"
    dataset_stego = r"..\data\raw\CFD_Dataset\CFD_Version_3_0\Images\CFD_stego"

    separar_stego(dataset_original, dataset_stego)
