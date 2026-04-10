# Presentación del Proyecto: Detector de Bicicletas

## 1. Resumen del Proyecto

Este proyecto consiste en un sistema de visión por computadora capaz de detectar y rastrear bicicletas en tiempo real a partir de una fuente de video (como un archivo o una cámara web).

El programa procesa el video cuadro a cuadro. En cada imagen, utiliza un modelo de red neuronal pre-entrenado para identificar la ubicación de las bicicletas. Una vez detectadas, dibuja un cuadro delimitador verde alrededor de cada una y muestra el video resultante en una ventana. Opcionalmente, puede guardar el video con las detecciones en un nuevo archivo.

## 2. Tecnologías Aplicadas

El detector se construye sobre un conjunto de librerías de Python especializadas en inteligencia artificial y visión por computadora:

*   **Python:** El lenguaje de programación principal sobre el que se construye toda la aplicación.
*   **Ultralytics YOLOv8:** Es la tecnología central del proyecto. Se utiliza una implementación de YOLO ("You Only Look Once"), una de las arquitecturas de redes neuronales más eficientes y precisas para la detección de objetos en tiempo real. El proyecto carga un modelo pre-entrenado (`yolov8n.pt`) que ya sabe cómo identificar 80 tipos de objetos comunes, incluida la bicicleta.
*   **PyTorch:** Es el framework de deep learning sobre el que se ejecuta el modelo YOLOv8. Se encarga de gestionar los cálculos complejos de la red neuronal, aprovechando la aceleración por hardware si está disponible.
*   **OpenCV (cv2):** Es una librería fundamental para la visión por computadora. En este proyecto, se utiliza para:
    *   Capturar el video desde un archivo (`salida_bicis.mp4`) o una cámara.
    *   Leer los cuadros (imágenes) del video uno por uno.
    *   Gestionar la visualización del video en una ventana.
    *   Escribir el video de salida con las detecciones ya dibujadas.

## 3. Uso de Redes Neuronales en el Proyecto

El "cerebro" de este detector es la red neuronal convolucional (CNN) conocida como **YOLOv8**. Así es como funciona dentro del flujo del programa:

1.  **Carga del Modelo:** Al iniciar, el programa carga el archivo `yolov8n.pt`. Este archivo contiene la arquitectura de la red YOLOv8 y, lo más importante, los "pesos" o el conocimiento que ha adquirido tras ser entrenada con millones de imágenes del dataset COCO. Este conocimiento le permite reconocer objetos.

2.  **Procesamiento por Cuadro:** El video no se analiza de una sola vez. Se descompone en una secuencia de imágenes (cuadros). Cada uno de estos cuadros se envía individualmente a la red neuronal.

3.  **Inferencia de la Red (La "Predicción"):** Para cada cuadro, la red YOLOv8 realiza una única pasada (de ahí "You Only Look Once") y produce una lista de todos los objetos que cree haber encontrado. Para cada objeto, la red proporciona:
    *   **Clase del Objeto:** La etiqueta de lo que es (ej: "persona", "coche", "bicicleta").
    *   **Cuadro Delimitador (Bounding Box):** Las coordenadas `(x, y, ancho, alto)` que definen un rectángulo alrededor del objeto.
    *   **Puntuación de Confianza (Confidence Score):** Un valor (de 0 a 1) que indica qué tan segura está la red de que su detección es correcta.

4.  **Filtrado de Resultados:** El código recibe todas las detecciones del modelo, pero solo le interesan las bicicletas. Por lo tanto, filtra la lista para quedarse únicamente con los objetos cuya clase sea **"bicicleta"** (que corresponde al ID `1` en el dataset COCO) y que tengan una puntuación de confianza superior a un umbral definido (por defecto, 0.5).

5.  **Visualización:** Una vez filtradas las detecciones, el programa utiliza las funciones de la librería `ultralytics` para dibujar los cuadros delimitadores y las etiquetas directamente sobre la imagen del cuadro.

Este ciclo de `leer cuadro -> inferencia -> filtrar -> dibujar` se repite para todo el video, creando la ilusión de una detección en tiempo real.
