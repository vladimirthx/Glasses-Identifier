import sys
import cv2
import numpy as np
import joblib
from PIL import Image

from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton,
    QLabel, QFileDialog, QVBoxLayout, QComboBox
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt

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
        self.setGeometry(100, 100, 400, 700)

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

        self.label_imagen = QLabel("Aquí se mostrará el rostro detectado")
        self.label_imagen.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label_resultado = QLabel("")
        self.label_resultado.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_resultado.setStyleSheet("font-size: 14px; font-weight: bold;")

        self.boton_cargar = QPushButton("Subir Imagen")
        self.boton_cargar.clicked.connect(self.cargar_imagen)

        self.label_resumen = QLabel("Resumen de predicciones:\n(Sube una imagen para ver los resultados)")
        self.label_resumen.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.label_resumen.setStyleSheet(
            "color: black; font-size: 12px; border: 1px solid gray; padding: 5px; background-color: #f0f0f0;")

        self.layout.addWidget(self.label_instruccion)
        self.layout.addWidget(self.combo_modelos)
        self.layout.addWidget(self.label_imagen)
        self.layout.addWidget(self.boton_cargar)
        self.layout.addWidget(self.label_resultado)
        self.layout.addWidget(self.label_resumen)

        self.setLayout(self.layout)

    def cargar_imagen(self):
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
                    rostro_gris_contiguo.tobytes(), width, height, bytes_per_line, QImage.Format.Format_Grayscale8
                )

                pixmap = QPixmap.fromImage(q_img)
                self.label_imagen.setPixmap(pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio))

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

            except Exception as e:
                self.label_resultado.setText(str(e))
                self.label_resultado.setStyleSheet("color: black;")
                self.label_resumen.setText("Error al procesar la imagen.")

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


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = Ventana()
    ventana.show()
    sys.exit(app.exec())