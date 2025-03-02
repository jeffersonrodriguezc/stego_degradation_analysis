from pathlib import Path

def organizar_imagenes(ruta_base: str):
    base_path = Path(ruta_base)

    for file_path in base_path.iterdir():
        # Ignorar si es directorio o archivo del sistema
        if file_path.is_dir() or file_path.name.startswith('.') or file_path.name == '.DS_Store':
            continue

        # Nombre del archivo sin extensión
        nombre_archivo = file_path.stem  # Ejemplo: "CFD-IF-601-519-N_stego"
        # Extensión del archivo (p.ej. .jpg, .png, etc.)
        extension = file_path.suffix

        # Quitar el sufijo "_stego" si está presente
        nombre_sin_stego = nombre_archivo.replace('_stego', '')  # "CFD-IF-601-519-N"

        # Separar por guiones
        partes = nombre_sin_stego.split('-')
        # Suponiendo que las primeras 4 partes forman el ID (por ej. "CFD-IF-601-519")
        folder_name = '-'.join(partes[1:3])

        # Crear la carpeta destino si no existe
        carpeta_destino = base_path / folder_name
        carpeta_destino.mkdir(exist_ok=True)

        # Mover el archivo a la carpeta
        destino = carpeta_destino / file_path.name
        file_path.rename(destino)


if __name__ == "__main__":
    # Puedes poner '.' para indicar la carpeta actual, o una ruta absoluta o relativa
    ruta_imagenes = r"..\data\raw\CFD_Dataset\CFD_Version_3_0\Images\CFD-INDIA"
    organizar_imagenes(ruta_imagenes)
