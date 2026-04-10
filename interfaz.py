import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from detector import OjectDetector

class DetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detección de Bicicletas - YOLOv8")
        self.root.geometry("460x400")
        self.root.resizable(False, False)
        self.root.configure(bg="#ecf0f1")

        # Cerrar app con la tecla Q
        self.root.bind("<q>", self.close_app)
        self.root.bind("<Q>", self.close_app)

        # Inicializar detector
        self.det = OjectDetector(model_path="yolov8n.pt")

        # ================== CONTENEDOR PRINCIPAL ==================
        container = tk.Frame(root, bg="#ecf0f1")
        container.pack(expand=True, fill="both", padx=20, pady=20)

        # ================== HEADER ==================
        header = tk.Frame(container, bg="#2c3e50", height=80)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="🚲 DETECCIÓN DE BICICLETAS",
            font=("Segoe UI", 16, "bold"),
            bg="#2c3e50",
            fg="white"
        ).pack(pady=(15, 0))

        tk.Label(
            header,
            text="Sistema con YOLOv8",
            font=("Segoe UI", 9),
            bg="#2c3e50",
            fg="#bdc3c7"
        ).pack()

        # ================== PANEL ==================
        panel = tk.Frame(container, bg="white", bd=0, relief="flat")
        panel.pack(fill="both", expand=True, pady=15)

        # ================== BOTONES ==================
        self.create_button(
            panel,
            text="📁 Detectar desde Video",
            color="#3498db",
            command=self.run_video
        )

        self.create_button(
            panel,
            text="🖼️ Detectar desde Imagen",
            color="#2ecc71",
            command=self.run_image
        )

        self.create_button(
            panel,
            text="📷 Cámara en Vivo",
            color="#e67e22",
            command=self.run_camera
        )

        # ================== INFO ==================
        tk.Label(
            panel,
            text="ℹ️ Presiona 'Q' para cerrar el sistema\nDurante la detección presiona 'Q' para salir",
            font=("Segoe UI", 9),
            bg="white",
            fg="#7f8c8d",
            justify="center"
        ).pack(pady=20)

    # ================== BOTÓN PERSONALIZADO ==================
    def create_button(self, parent, text, color, command):
        btn = tk.Button(
            parent,
            text=text,
            font=("Segoe UI", 11, "bold"),
            bg=color,
            fg="white",
            activebackground=color,
            activeforeground="white",
            relief="flat",
            cursor="hand2",
            height=2,
            command=command
        )
        btn.pack(fill="x", pady=8, padx=30)

    # ================== FUNCIONES ==================
    def run_video(self):
        path = filedialog.askopenfilename(
            title="Selecciona un video",
            filetypes=[("Videos", "*.mp4 *.avi *.mov")]
        )
        if path:
            self.det.detect_video_bicycles(
                path,
                out_path="salida_video.mp4",
                show=True
            )

    def run_image(self):
        path = filedialog.askopenfilename(
            title="Selecciona una imagen",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png")]
        )
        if path:
            self.det.detect_image_bicycles(
                path,
                out_path="resultado_bici.jpg",
                show=True
            )

    def run_camera(self):
        messagebox.showinfo(
            "Cámara",
            "Cámara iniciada.\nPresiona 'Q' para cerrar."
        )
        self.det.detect_video_bicycles(
            0,
            out_path="captura_camara.mp4",
            show=True
        )

    # ================== CERRAR APP ==================
    def close_app(self, event=None):
        if messagebox.askyesno("Salir", "¿Deseas cerrar el sistema?"):
            self.root.destroy()

# ================== MAIN ==================
if __name__ == "__main__":
    root = tk.Tk()
    app = DetectorApp(root)
    root.mainloop()
