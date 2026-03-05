import sys
import cv2
import numpy as np
import joblib
from PIL import Image

from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton,
    QLabel, QFileDialog, QVBoxLayout, QComboBox, QHBoxLayout
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QTimer

# =====================================================
# CONFIGURACIÓN DETECTOR FACIAL
# =====================================================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


# =====================================================
# FUNCIONES DE PREPROCESAMIENTO
# =====================================================

def imagen_valida(ruta):
    try:
        img = Image.open(ruta)
        img.verify()
        return True
    except:
        return False


def detectar_y_recortar_rostro(ruta):
    imagen = cv2.imread(ruta)
    if imagen is None: return None
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    gris = cv2.equalizeHist(gris)

    rostros = face_cascade.detectMultiScale(gris, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in rostros:
        margen_y = int(h * 0.15)
        margen_x = int(w * 0.15)

        y1 = max(0, y - margen_y)
        y2 = min(gris.shape[0], y + h + margen_y)
        x1 = max(0, x - margen_x)
        x2 = min(gris.shape[1], x + w + margen_x)

        rostro = gris[y1:y2, x1:x2]
        return rostro
    return None


def preprocesar_rostro(rostro, tamaño=(64, 64)):
    rostro_res = cv2.resize(rostro, tamaño)
    rostro_norm = rostro_res.astype("float32") / 255.0
    rostro_plano = rostro_norm.flatten().reshape(1, -1)
    return rostro_plano


# =====================================================
# INTERFAZ PYQT6
# =====================================================

class Ventana(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Detector de Lentes - Multimodelo")
        self.setGeometry(100, 100, 400, 800)

        # Estado de la cámara
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._actualizar_frame_camara)
        self.modo_camara = False

        self.layout = QVBoxLayout()

        self.label_instruccion = QLabel("Selecciona el modelo a visualizar a detalle:")
        self.combo_modelos = QComboBox()

        self.diccionario_modelos = {
            "Red Neuronal (MLP)": "modelo_neural_network.pkl",
            "Máquinas de Vectores de Soporte (SVM)": "modelo_svm.pkl",
            "Random Forest": "modelo_random_forest.pkl",
            "Regresión Logística": "modelo_logistic_regression.pkl",
            "K-Nearest Neighbors (KNN)": "modelo_knn.pkl"
        }

        self.combo_modelos.addItems(self.diccionario_modelos.keys())
        self.combo_modelos.currentIndexChanged.connect(self.actualizar_resultado)

        self.resultados_prediccion = {}

        # Área de imagen / preview cámara 
        self.label_imagen = QLabel("Aquí se mostrará el rostro detectado")
        self.label_imagen.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_imagen.setMinimumHeight(300)
        self.label_imagen.setStyleSheet("background-color: #1a1a1a; color: white;")

        self.label_resultado = QLabel("")
        self.label_resultado.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_resultado.setStyleSheet("font-size: 14px; font-weight: bold;")

        # Fila de botones principales
        fila_botones = QHBoxLayout()
        self.boton_cargar = QPushButton("Subir Imagen")
        self.boton_cargar.clicked.connect(self.cargar_imagen)

        self.boton_camara = QPushButton("Usar Cámara")
        self.boton_camara.clicked.connect(self.alternar_camara)

        fila_botones.addWidget(self.boton_cargar)
        fila_botones.addWidget(self.boton_camara)

        # Botón tomar foto (oculto hasta activar cámara)
        self.boton_capturar = QPushButton("Tomar Foto y Analizar")
        self.boton_capturar.clicked.connect(self.tomar_foto)
        self.boton_capturar.setVisible(False)

        self.label_resumen = QLabel("Resumen de predicciones:\n(Sube una imagen para ver los resultados)")
        self.label_resumen.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.label_resumen.setStyleSheet(
            "color: black; font-size: 12px; border: 1px solid gray; padding: 5px; background-color: #f0f0f0;")

        self.label_consenso = QLabel("")
        self.label_consenso.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_consenso.setWordWrap(True)
        self.label_consenso.setStyleSheet(
            "font-size: 13px; font-weight: bold; border: 2px solid #444; "
            "padding: 8px; background-color: #e8e8e8; border-radius: 6px;"
        )

        self.layout.addWidget(self.label_instruccion)
        self.layout.addWidget(self.combo_modelos)
        self.layout.addWidget(self.label_imagen)
        self.layout.addLayout(fila_botones)
        self.layout.addWidget(self.boton_capturar)
        self.layout.addWidget(self.label_resultado)
        self.layout.addWidget(self.label_resumen)
        self.layout.addWidget(self.label_consenso)

        self.setLayout(self.layout)

    # =====================================================
    # CÁMARA INLINE
    # =====================================================

    def alternar_camara(self):
        if not self.modo_camara:
            self._iniciar_camara()
        else:
            self._detener_camara()

    def _iniciar_camara(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.label_resultado.setText("No se pudo abrir la cámara.")
            return

        self.modo_camara = True
        self.boton_camara.setText("Detener Cámara")
        self.boton_capturar.setVisible(True)
        self.label_resultado.setText("Cámara activa — colócate frente a ella")
        self.timer.start(30)

    def _detener_camara(self):
        self.timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None
        self.modo_camara = False
        self.boton_camara.setText("Usar Cámara")
        self.boton_capturar.setVisible(False)
        self.label_imagen.clear()
        self.label_imagen.setText("Aquí se mostrará el rostro detectado")
        self.label_resultado.setText("")

    def _actualizar_frame_camara(self):
        if not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        # Dibujar rectángulos sobre rostros detectados en tiempo real
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostros = face_cascade.detectMultiScale(gris, scaleFactor=1.2, minNeighbors=5)
        for (x, y, w, h) in rostros:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h_f, w_f, ch = frame_rgb.shape
        q_img = QImage(frame_rgb.data, w_f, h_f, ch * w_f, QImage.Format.Format_RGB888)
        self.label_imagen.setPixmap(
            QPixmap.fromImage(q_img).scaled(
                self.label_imagen.width(), self.label_imagen.height(),
                Qt.AspectRatioMode.KeepAspectRatio
            )
        )

    # =====================================================
    # TOMAR FOTO DESDE CÁMARA
    # =====================================================

    def tomar_foto(self):
        if not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            self.label_resultado.setText("Error al capturar el frame.")
            return

        # Pausar preview mientras procesamos
        self.timer.stop()

        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gris_eq = cv2.equalizeHist(gris)
        rostros = face_cascade.detectMultiScale(gris_eq, scaleFactor=1.2, minNeighbors=5)

        if len(rostros) == 0:
            self.label_resultado.setText("⚠️ No se detectó ningún rostro.\nIntenta de nuevo.")
            self.label_resultado.setStyleSheet("color: orange; font-size: 13px; font-weight: bold;")
            self.timer.start(30)  # Reanudar preview
            return

        # Recortar primer rostro detectado
        (x, y, w, h) = rostros[0]
        margen_y = int(h * 0.15)
        margen_x = int(w * 0.15)
        y1 = max(0, y - margen_y)
        y2 = min(gris_eq.shape[0], y + h + margen_y)
        x1 = max(0, x - margen_x)
        x2 = min(gris_eq.shape[1], x + w + margen_x)
        rostro_gris = gris_eq[y1:y2, x1:x2]

        # Mostrar rostro capturado en el mismo label_imagen
        rh, rw = rostro_gris.shape
        rostro_contiguo = np.ascontiguousarray(rostro_gris)
        q_rostro = QImage(rostro_contiguo.tobytes(), rw, rh, rw, QImage.Format.Format_Grayscale8)
        self.label_imagen.setPixmap(
            QPixmap.fromImage(q_rostro).scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio)
        )

        # Ejecutar modelos
        self._ejecutar_modelos(rostro_gris)

        # Reanudar preview
        self.timer.start(30)

    # =====================================================
    # SUBIR IMAGEN
    # =====================================================

    def cargar_imagen(self):
        # Si la cámara está activa, la detenemos primero
        if self.modo_camara:
            self._detener_camara()

        ruta, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar imagen", "", "Images (*.png *.jpg *.jpeg)"
        )

        if ruta:
            try:
                if not imagen_valida(ruta):
                    raise ValueError("Imagen inválida o corrupta")

                rostro_gris = detectar_y_recortar_rostro(ruta)
                if rostro_gris is None:
                    raise ValueError("No se detectó ningún rostro en la imagen")

                height, width = rostro_gris.shape
                bytes_per_line = width
                rostro_gris_contiguo = np.ascontiguousarray(rostro_gris)

                q_img = QImage(
                    rostro_gris_contiguo.tobytes(), width, height, bytes_per_line,
                    QImage.Format.Format_Grayscale8
                )

                pixmap = QPixmap.fromImage(q_img)
                self.label_imagen.setPixmap(pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio))

                self._ejecutar_modelos(rostro_gris)

            except Exception as e:
                self.label_resultado.setText(str(e))
                self.label_resultado.setStyleSheet("color: black;")
                self.label_resumen.setText("Error al procesar la imagen.")

    # =====================================================
    # LÓGICA: EJECUTAR MODELOS + CONSENSO
    # =====================================================

    def _ejecutar_modelos(self, rostro_gris):
        input_modelo = preprocesar_rostro(rostro_gris)

        self.resultados_prediccion.clear()
        texto_resumen = "Resumen de predicciones:\n\n"

        for nombre, archivo in self.diccionario_modelos.items():
            try:
                modelo_cargado = joblib.load(archivo)
                prediccion = modelo_cargado.predict(input_modelo)
                probabilidad = modelo_cargado.predict_proba(input_modelo)

                self.resultados_prediccion[nombre] = {
                    "prediccion": prediccion[0],
                    "probabilidad": np.max(probabilidad) * 100
                }

                veredicto = "LENTES" if prediccion[0] == 1 else "SIN LENTES"
                texto_resumen += f"• {nombre}: {veredicto} ({np.max(probabilidad) * 100:.1f}%)\n"

            except Exception as e:
                print(f"Error detallado en {nombre}: {e}")
                self.resultados_prediccion[nombre] = {"error": str(e)}
                texto_resumen += f"• {nombre}: Error al cargar\n"

        self.label_resumen.setText(texto_resumen)
        self.actualizar_resultado()
        self.mostrar_consenso()

    def mostrar_consenso(self):
        if not self.resultados_prediccion:
            return

        votos_lentes = sum(
            1 for r in self.resultados_prediccion.values()
            if "error" not in r and r["prediccion"] == 1
        )
        votos_sin = sum(
            1 for r in self.resultados_prediccion.values()
            if "error" not in r and r["prediccion"] == 0
        )
        total = votos_lentes + votos_sin

        if total == 0:
            self.label_consenso.setText("No se pudo calcular el consenso.")
            return

        veredicto = "TIENE LENTES" if votos_lentes > votos_sin else "NO TIENE LENTES"
        color_fondo = "#c8f7c5" if votos_lentes > votos_sin else "#f7c5c5"
        color_borde = "#2ecc71" if votos_lentes > votos_sin else "#e74c3c"

        texto = (
            f"━━━ VOTACIÓN FINAL ━━━\n"
            f"Con lentes: {votos_lentes} modelo(s)  |  Sin lentes: {votos_sin} modelo(s)\n"
            f"Decisión por mayoría: {veredicto}"
        )
        self.label_consenso.setText(texto)
        self.label_consenso.setStyleSheet(
            f"font-size: 13px; font-weight: bold; border: 2px solid {color_borde}; "
            f"padding: 8px; background-color: {color_fondo}; border-radius: 6px;"
        )

    def actualizar_resultado(self):
        if not self.resultados_prediccion:
            return

        nombre_seleccionado = self.combo_modelos.currentText()
        resultado = self.resultados_prediccion.get(nombre_seleccionado)

        if "error" in resultado:
            self.label_resultado.setText(f"Modelo: {nombre_seleccionado}\nError al cargar o predecir.")
            self.label_resultado.setStyleSheet("color: orange; font-weight: bold; font-size: 14px;")
            return

        if resultado["prediccion"] == 1:
            self.label_resultado.setText(
                f"Modelo: {nombre_seleccionado}\nResultado: LENTES ({resultado['probabilidad']:.2f}%)")
            self.label_resultado.setStyleSheet("color: green; font-weight: bold; font-size: 14px;")
        else:
            self.label_resultado.setText(
                f"Modelo: {nombre_seleccionado}\nResultado: SIN LENTES ({resultado['probabilidad']:.2f}%)")
            self.label_resultado.setStyleSheet("color: red; font-weight: bold; font-size: 14px;")

    def closeEvent(self, event):
        self._detener_camara()
        event.accept()


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = Ventana()
    ventana.show()
    sys.exit(app.exec())
