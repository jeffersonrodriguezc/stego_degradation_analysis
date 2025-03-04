import os
import cv2
import numpy as np
import pytest
import matplotlib.pyplot as plt
from core.transformations import Transformations

def load_test_image(image_path):
    """
    Carga una imagen en formato OpenCV desde la ruta indicada.
    """
    image = cv2.imread(image_path)
    assert image is not None, f"Error: no se pudo cargar la imagen desde {image_path}"
    return image


def test_single_image_format_consistency():
    """
    Prueba que, al aplicar una transformación a una imagen en formato OpenCV,
    la imagen de salida siga siendo un array NumPy (OpenCV).
    """
    # Suponiendo que tienes una imagen de prueba en la carpeta 'datos'
    image_path = os.path.join("datos", "test_image.jpg")
    img = load_test_image(image_path)
    
    transformer = Transformations()
    # Por ejemplo, aplicar crop con zoom_factor=1.0 (sin efecto real)
    transformed = transformer.apply_crop(img, zoom_factor=1.0)
    
    # Verificar que la salida es un array NumPy (formato OpenCV)
    assert isinstance(transformed, np.ndarray), "La imagen de salida no es del tipo OpenCV (numpy.ndarray)"
    
def test_multiple_images_parallel():
    """
    Prueba que al pasar una lista de imágenes a una transformación,
    la salida sea una lista de imágenes en formato OpenCV.
    """
    image_path = os.path.join("datos", "test_image.jpg")
    img = load_test_image(image_path)
    images = [img, img]  # Simulamos un dataset con dos imágenes iguales
    
    transformer = Transformations()
    # Aplicamos, por ejemplo, la transformación de resize
    results = transformer.apply_transformation(images, 'resize', scale=0.9, target_size=(224, 224))
    
    # Verificar que se retorna una lista y que cada imagen es un array NumPy
    assert isinstance(results, list), "La salida debe ser una lista de imágenes."
    for out_img in results:
        assert isinstance(out_img, np.ndarray), "Cada imagen de salida debe ser en formato OpenCV (numpy.ndarray)."
