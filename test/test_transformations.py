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

def visualize_side_by_side(img1, img2, title1="Original", title2="Transformed"):
    """
    Muestra dos imágenes lado a lado usando matplotlib.
    Se convierte la imagen de BGR (OpenCV) a RGB para visualización.
    """
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img1_rgb)
    axs[0].set_title(title1)
    axs[0].axis("off")
    axs[1].imshow(img2_rgb)
    axs[1].set_title(title2)
    axs[1].axis("off")
    plt.show()

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

def test_visualization_side_by_side():
    """
    Prueba auxiliar para visualizar la imagen original y la transformada lado a lado.
    Esta prueba se ejecuta manualmente para inspeccionar visualmente el resultado.
    """
    image_path = os.path.join("datos", "test_image.jpg")
    img = load_test_image(image_path)
    
    transformer = Transformations()
    # Aplicamos una transformación de ejemplo (por ejemplo, brightness/contrast)
    transformed = transformer.apply_brightness_contrast(img, alpha=0.8, beta=30)
    
    # Visualización side by side
    visualize_side_by_side(img, transformed, "Original", "Brillo/Contraste Ajustados")
    
    # Finalizamos la prueba sin errores (solo visual)
    assert True
