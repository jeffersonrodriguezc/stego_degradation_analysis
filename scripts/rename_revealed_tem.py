import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Función para renombrar un único archivo PNG
def rename_file(root, file):
    if file.endswith('.png') and file.startswith('revealed_secret_'):
        old_filepath = os.path.join(root, file)

        new_filename = file.replace('revealed_secret_stego_', '', 1).replace('_original_224_224.png', '_recovered.png')
        new_filepath = os.path.join(root, new_filename)

        os.rename(old_filepath, new_filepath)

        print(f'Renamed: {old_filepath} -> {new_filepath}')
        return new_filepath
    return None

# Función principal para renombrar y recolectar archivos PNG
def rename_png_files_and_generate_csv(base_folder, csv_output_path):
    output_images = []

    num_cpus = multiprocessing.cpu_count() // 2
    with ThreadPoolExecutor(max_workers=num_cpus) as executor:
        futures = []

        for root, dirs, files in os.walk(base_folder):
            for file in files:
                futures.append(executor.submit(rename_file, root, file))

        for future in futures:
            result = future.result()
            if result:
                output_images.append(result)

    # Manejo del archivo CSV
    if os.path.exists(csv_output_path):
        existing_df = pd.read_csv(csv_output_path)
        new_df = pd.DataFrame({'output_image': output_images})
        df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        df = pd.DataFrame({'output_image': output_images})

    df.to_csv(csv_output_path, index=False)
    print(f"CSV generado/actualizado con éxito en: {csv_output_path}")

# Ejemplo de uso
base_folder = "/app/output/reveal/CFD_one_shot/CFD_one_shot_stegformer/stego" 
csv_output_path = '/app/output/reveal/CFD_one_shot/output_images.csv'

rename_png_files_and_generate_csv(base_folder, csv_output_path)
