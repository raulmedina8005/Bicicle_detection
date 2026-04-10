import ultralytics
from ultralytics import YOLO
import numpy as np
import cv2

class ErrorDetector(Exception):
    pass

class OjectDetector:
    def __init__(self, model_path="yolov8n.pt", model_format="onnx"):
        self.model_path = model_path
        self.model_format = model_format
        
        self.INPUT_WIDTH = 640
        self.INPUT_HEIGHT = 640
        self.CONFIDENCE_THRESHOLD = 0.5
        self.NMS_THRESHOLD = 0.4
        
        self.FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SCALE = 1
        self.THICKNESS = 2
        
    def load_model(self):
        try:
            model = YOLO(self.model_path)
            model.export(format=self.model_format)
        except:
            raise ErrorDetector("Error loading or exporting the model.")
        return model
    
    def load_class_names(self):
        with open("./tools/coco.names", "r") as f:
            class_names = f.read().splitlines()
        return class_names
    
    def draw_detections(self, img, box, score, class_id):

        # Extraer las coordenadas del cuadro delimitador
        x1, y1, w, h = box

        # Recuperar el color para el ID de clase
        color = (0, 255, 0)

        # Dibuja el cuadro delimitador en la imagen
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Crea el texto de la etiqueta con el nombre de la clase y la puntuación.
        label = f"{self.load_class_names()[class_id]}: {score:.2f}"

        # Calcular las dimensiones del texto de la etiqueta
        (label_width, label_height), _ = cv2.getTextSize(label, self.FONT_FACE, self.FONT_SCALE, self.THICKNESS)

        # Calcular la posición del texto de la etiqueta
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Dibuje un rectángulo relleno como fondo para el texto de la etiqueta.
        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )

        # Dibuja el texto de la etiqueta en la imagen.
        cv2.putText(img, label, (label_x, label_y), self.FONT_FACE, self.FONT_SCALE, (0, 0, 0), self.THICKNESS, cv2.LINE_AA)
        
    def inference(self, input_image, net):

        # Crea un blob 4D a partir de un marco.
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (self.INPUT_WIDTH, self.INPUT_HEIGHT), [0,0,0], 1, crop=False)

        # Establece la entrada a la red.
        net.setInput(blob)

        # Ejecuta el pase hacia adelante para obtener la salida de las capas de salida.
        output_layers = net.getUnconnectedOutLayersNames()
        image_data = net.forward(output_layers)

        # Devolver los datos de la imagen preprocesada
        return image_data
    
    def detect_video_bicycles(
    self,
    video_source,
    out_path=None,
    show=True,
    imgsz=640,
    target_class_id=1,          # COCO: bicycle
    conf_thres=None,
    quit_key="q"
):

        if conf_thres is None:
            conf_thres = self.CONFIDENCE_THRESHOLD

        # Abrir video/cámara
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise ErrorDetector(f"No se pudo abrir la fuente de video: {video_source}")

        try:
            if str(self.model_path).lower().endswith(".onnx"):
                model = YOLO(self.model_path, task="detect")
            else:
                model = YOLO(self.model_path)
        except Exception as e:
            cap.release()
            raise ErrorDetector(f"Error cargando el modelo: {e}")

        # Preparar writer (opcional)
        writer = None
        if out_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps is None or fps == 0:
                fps = 30.0  # fallback razonable
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            if not writer.isOpened():
                cap.release()
                raise ErrorDetector(f"No se pudo crear el archivo de salida: {out_path}")

        window_name = "YOLO - SOLO BICICLETAS"
        if show:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 960, 540)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(
                    frame,
                    imgsz=imgsz,
                    conf=conf_thres,
                    verbose=False
                )

                if results and len(results) > 0:
                    r = results[0]

                    if r.boxes is not None and len(r.boxes) > 0:
                        cls = r.boxes.cls  # tensor
                        keep = (cls == target_class_id)

                        if keep.sum().item() > 0:
                            r.boxes = r.boxes[keep]
                            annotated = r.plot()
                        else:
                            annotated = frame
                    else:
                        annotated = frame
                else:
                    annotated = frame

                if writer is not None:
                    writer.write(annotated)

                if show:
                    cv2.imshow(window_name, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(quit_key):
                        break

            return True

        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if show:
                cv2.destroyWindow(window_name)


    def detect_image_bicycles(self, img_path, out_path=None, show=True, target_class_id=1):
        # 1. Cargar la imagen
        frame = cv2.imread(img_path)
        if frame is None:
            raise ErrorDetector(f"No se pudo cargar la imagen en: {img_path}")

        # 2. Cargar el modelo
        model = YOLO(self.model_path)

        # 3. Inferencia
        results = model(frame, imgsz=self.INPUT_WIDTH, conf=self.CONFIDENCE_THRESHOLD, verbose=False)

        # 4. Filtrar por clase (bicicletas)
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                mask = (r.boxes.cls == target_class_id)
                if mask.sum().item() > 0:
                    r.boxes = r.boxes[mask]
                    annotated = r.plot()
                else:
                    annotated = frame
            else:
                annotated = frame
        else:
            annotated = frame

        # 5. Guardar resultado
        if out_path:
            cv2.imwrite(out_path, annotated)

        # 6. Mostrar resultado
        if show:
            cv2.imshow("Deteccion Imagen", annotated)
            cv2.waitKey(0) # Espera a presionar una tecla
            cv2.destroyAllWindows()

        return annotated
