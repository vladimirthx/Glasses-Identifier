# Importe de bibliotecas
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import joblib
import pandas as pd

# Obtener el dataset
faces = fetch_olivetti_faces()
X = faces.data

# Asignación de etiquetas manuales
y_glasses = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y_glasses, test_size=0.2, random_state=42, stratify=y_glasses
)

if __name__ == "__main__":

    # 1. Definir los 5 modelos
    modelos = {
        "Neural_Network": MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu', solver='adam',
                                        max_iter=1000, random_state=42),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

    # Se genera un DataFrame para almacenar todas las matrices de confusión
    all_confusion_matrices = []

    # 2. Entrenar, evaluar y guardar cada uno
    for nombre, modelo in modelos.items():
        print(f"\nEntrenando {nombre}...")
        modelo.fit(X_train, y_train)

        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # Calcular matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        print(f"Matriz de Confusión:\n{cm}")

        print(f"Precisión (Accuracy): {acc:.4f}")

        # Guardar el archivo .pkl con el nombre del modelo
        nombre_archivo = f'modelo_{nombre.lower()}.pkl'
        joblib.dump(modelo, nombre_archivo)
        print(f"Guardado como: {nombre_archivo}")
        
        # Crear un DataFrame para esta matriz de confusión
        cm_df = pd.DataFrame(cm, 
                           index=['Real: Sin Gafas', 'Real: Con Gafas'],
                           columns=['Pred: Sin Gafas', 'Pred: Con Gafas'])
        
        # Añadir columna con el nombre del modelo
        cm_df.insert(0, 'Modelo', nombre)
        cm_df.insert(1, 'Métrica', 'Matriz de Confusión')
        
        all_confusion_matrices.append(cm_df)

    # Combinar todas las matrices en un solo DataFrame
    final_df = pd.concat(all_confusion_matrices, ignore_index=True)
    
    # Exportar a CSV
    csv_filename = 'matrices_confusion_latex.csv'
    final_df.to_csv(csv_filename, index=False, encoding='utf-8')
    print(f"\nMatrices de confusión exportadas a: {csv_filename}")
    
    # Compactar para facilitar la importación en LaTeX
    compact_data = []
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        compact_data.append({
            'Modelo': nombre,
            'TN': cm[0, 0],  # Verdaderos Negativos (sin gafas correctamente identificados)
            'FP': cm[0, 1],  # Falsos Positivos
            'FN': cm[1, 0],  # Falsos Negativos
            'TP': cm[1, 1]   # Verdaderos Positivos (con gafas correctamente identificados)
        })
    
    compact_df = pd.DataFrame(compact_data)
    compact_csv = 'matrices_confusion_compacto.csv'
    compact_df.to_csv(compact_csv, index=False, encoding='utf-8')
    print(f"Versión compacta exportada a: {compact_csv}")

    print("\n¡Todos los modelos han sido entrenados y guardados exitosamente!")