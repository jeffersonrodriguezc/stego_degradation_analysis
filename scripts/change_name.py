from pathlib import Path

def renombrar_stego(carpeta_base: str):
    # Convertir a Path absoluto para evitar problemas de ruta
    base_path = Path(carpeta_base).resolve()
    
    # Recorrer todos los archivos en todas las subcarpetas
    for file_path in base_path.rglob("*"):
        if file_path.is_file():
            # Si el nombre del archivo termina con 'stego.png'
            if file_path.name.endswith("stego.png"):
                # Generar el nuevo nombre reemplazando 'stego.png' por 'stego_1.png'
                nuevo_nombre = file_path.name.replace("stego.png", "stego_1.png")
                nuevo_path = file_path.with_name(nuevo_nombre)
                print(f"Renombrando: {file_path} -> {nuevo_path}")
                # Renombrar el archivo
                file_path.rename(nuevo_path)

if __name__ == "__main__":
    # Solicitar al usuario la ruta de la carpeta a procesar
    ruta_carpeta = "..\data\processed\CFD"
    renombrar_stego(ruta_carpeta)
