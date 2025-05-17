import os
import sys
import pandas as pd
import numpy as np
import multiprocessing
from joblib import Parallel, delayed

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from core.metrics import Metrics
from core.utils import load_image

# Path de la imagen secreta que no cambia
SECRET_IMAGE_PATH = "/app/data/secret/base/base_secret.jpg"
SECRET_IMAGE = load_image(SECRET_IMAGE_PATH)

# Cargar DataFrame original
csv_path = '/app/output/reveal/CFD_one_shot/output_images.csv'
df = pd.read_csv(csv_path)

# Función para calcular métricas por fila
def calculate_metrics(row, secret_image):
    # Cargar imagen recuperada
    output_image = load_image(row['output_image'])
    
    # Asegurar formato correcto (0-255)
    recovered_secret = output_image

    evaluator_reveal = Metrics(secret_image, recovered_secret)
    metrics = evaluator_reveal.compute_all()
    
    # Convertir diccionario a string para almacenar en DataFrame
    return str(metrics)

# Usar el 60% de núcleos disponibles
total_cores = multiprocessing.cpu_count()
n_jobs = max(1, int(total_cores * 0.6))

# Ejecución paralela de cálculo de métricas
df['metrics'] = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(calculate_metrics)(row, SECRET_IMAGE) for _, row in df.iterrows()
)

# Guardar el DataFrame actualizado
df.to_csv('/app/output/reveal/CFD_one_shot/dataframe_con_metricas.csv', index=False)

print("Cálculo de métricas completado y almacenado en 'dataframe_con_metricas.csv'.")
