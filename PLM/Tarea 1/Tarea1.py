import cv2
import numpy as np

# Función para procesar una imagen
def process_image(image_path):
    # Cargar la imagen
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"No se pudo cargar la imagen: {image_path}")
        return None, None, None
    
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Binarización con threshold
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Crear un kernel para dilate y erode
    kernel = np.ones((5, 5), np.uint8)
    
    # Aplicar dilate y luego erode
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    
    return image, thresh, eroded

# Función principal para mostrar las imágenes
def display_images():
    image_index = 1
    
    while True:
        # Ruta de la imagen
        image_path = f"imagen_{image_index}.png"
        
        # Procesar la imagen
        image, thresh, morphed = process_image(image_path)
        
        if image is None:
            print(f"No se encontró la imagen: {image_path}. Saliendo del programa.")
            break
        
        # Mostrar las imágenes
        cv2.imshow("Imagen Original", image)
        cv2.imshow("Binarización con Threshold", thresh)
        cv2.imshow("Morfología (Dilate y Erode)", morphed)
        
        # Esperar a que se presione Enter
        key = cv2.waitKey(0)
        if key == 13:  # Enter
            image_index += 1  # Pasar a la siguiente imagen
            cv2.destroyAllWindows()  # Cerrar las ventanas antes de cargar la nueva imagen
        else:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    display_images()
