import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# ✅ Creo los datos de empleados
datos = {
    "antiguedad": [
        1, 2, 3, 4, 5, 6, 6, 7, 8, 9,
        10, 2, 3, 4, 5, 7, 9, 1, 3, 6,
        8, 10, 2, 3, 5, 7, 8, 9, 6, 4
    ],
    "edad": [
        22, 35, 17, 29, 12, 15, 50, 30, 40, 38,
        45, 28, 25, 31, 33, 42, 48, 20, 24, 27,
        37, 39, 32, 21, 34, 44, 41, 46, 26, 29
    ],
    "area": [
        1, 6, 4, 2, 5, 1, 3, 2, 1, 3,
        5, 2, 4, 1, 2, 3, 5, 1, 4, 2,
        3, 5, 2, 1, 4, 3, 1, 2, 6, 1
    ],
    "ascendio": [
        1, 0, 0, 0, 0, 1, 1, 1, 1, 1,
        1, 0, 0, 0, 1, 1, 1, 0, 0, 1,
        1, 1, 0, 0, 1, 1, 1, 1, 0, 0
    ]
}

# ✅ Convierto en DataFrame
df = pd.DataFrame(datos)

# ✅ Separo variables predictoras y objetivo
X = df[["antiguedad", "edad", "area"]]
y = df["ascendio"]

# ✅ Divido en datos de entrenamiento y prueba
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ✅ Entreno modelos
modelo_log = LogisticRegression()
modelo_log.fit(X_entrenamiento, y_entrenamiento)

modelo_arbol = DecisionTreeClassifier()
modelo_arbol.fit(X_entrenamiento, y_entrenamiento)

# ✅ Guardo el modelo de árbol como archivo .pkl
joblib.dump(modelo_arbol, "modelo_ascensos.pkl")

# ✅ Servidor Flask para exponer el modelo
from flask import Flask, request, jsonify

app = Flask(__name__)
modelo = joblib.load("modelo_ascensos.pkl")

@app.route("/", methods=["POST"])
def predecir():
    datos = request.get_json()
    entrada = pd.DataFrame([{
    "antiguedad": datos["antiguedad"],
    "edad": datos["edad"],
    "area": datos["area"]
}])

    prediccion = modelo.predict(entrada)
    resultado = "Sí, este empleado podría ser ascendido" if prediccion[0] == 1 else "No, aún no parece listo"
    return jsonify({"resultado": resultado})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
