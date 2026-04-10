from detector import *

det = OjectDetector(model_path="yolov8n.pt")
det.detect_video_bicycles("../video5.mp4", out_path="salida_bicis.mp4", show=True)

# CAMARA DE LA COMPUTADORA
#det.detect_video_bicycles(0, out_path="salida_bicis.mp4", show=True)

# IMAGENES
#det.detect_image_bicycles("../img2.jpeg", out_path="resultado_bici.jpg", show=True)