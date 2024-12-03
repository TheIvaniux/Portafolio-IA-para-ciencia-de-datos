import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# Función para cerrar la ventana al presionar Enter
def cerrar_con_enter(event):
    if event.key == 'enter':  # Cerrar solo si se presiona Enter
        plt.close()

# Función para mostrar las imágenes con título
def mostrar_imagenes(image_original, image_filtrada):
    plt.figure(figsize=(10, 5))
    
    # Mostrar la imagen original
    plt.subplot(1, 2, 1)
    plt.imshow(image_original, cmap='gray')
    plt.title('Imagen Original')
    plt.xticks([]), plt.yticks([])

    # Mostrar la imagen con filtro de mediana
    plt.subplot(1, 2, 2)
    plt.imshow(image_filtrada, cmap='gray')
    plt.title('Mascara binaria')
    plt.xticks([]), plt.yticks([])

    # Conectar el evento para detectar la tecla 'Enter'
    plt.gcf().canvas.mpl_connect('key_press_event', cerrar_con_enter)

    plt.show()

# Inicializar el contador de imágenes
image_index = 1

while True:
    # Generar el nombre del archivo automáticamente
    image_path = f'cell{image_index}.png'

    # Verificar si el archivo existe
    if not os.path.exists(image_path):
        print(f"No se encontró el archivo: {image_path}")
        break

    # Cargar la imagen en escala de grises
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Paso 1: Suavizado con Filtro Gaussiano
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Paso 2: Detección de bordes usando el método Canny
    edges = cv2.Canny(blurred, threshold1=30, threshold2=100)

    # Paso 3: Umbralización con Otsu
    _, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Paso 4: Transformaciones morfológicas (apertura y cierre)
    kernel = np.ones((5, 5), np.uint8)
    morphed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # Paso 5: Aplicar erosión a la máscara final
    eroded_mask = cv2.erode(morphed, kernel, iterations=1)

    # Paso 6: Aplicar filtro de mediana 
    median_filtered = cv2.medianBlur(eroded_mask, 19)

    # Mostrar la imagen original y la imagen final filtrada
    mostrar_imagenes(image, median_filtered)

    # Incrementar el índice para la siguiente imagen
    image_index += 1
