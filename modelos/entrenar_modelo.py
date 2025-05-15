import pandas as pd
import unicodedata
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
from graphviz import Digraph

# Función para limpiar texto
def limpiar_texto(texto):
    # Pasar a minúsculas
    texto = texto.lower()
    # Eliminar tildes
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto)
                    if unicodedata.category(c) != 'Mn')
    # Eliminar signos de puntuación
    texto = re.sub(r'[^\w\s]', '', texto)
    # Eliminar espacios múltiples
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

# Cargar dataset
dataset = pd.read_csv("datos/data_set.csv", skipinitialspace=True)

# Limpiar preguntas
dataset['Pregunta_limpia'] = dataset['Pregunta'].apply(limpiar_texto)

# Separar preguntas limpias y respuestas
preguntas = dataset['Pregunta_limpia'].values
respuestas = dataset['Respuesta'].values

# Vectorizar
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(preguntas)

# Entrenar modelo
modelo = MultinomialNB()
modelo.fit(X, respuestas)

# Guardar modelo y vectorizer
joblib.dump(modelo, "modelo_entrenado.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Función de respuesta
def obtener_respuesta(pregunta_usuario):
    pregunta_limpia = limpiar_texto(pregunta_usuario)
    pregunta_vectorizada = vectorizer.transform([pregunta_limpia])
    probas = modelo.predict_proba(pregunta_vectorizada)[0]
    indice_max = probas.argmax()
    confianza = probas[indice_max]

    if confianza < 0.6:
     return "No entendi lo que me preguntaste, me lo podes repetir?"

    respuesta = modelo.classes_[indice_max]
    return respuesta

dot = Digraph()

dot.attr(rankdir='LR', size='8,5')

dot.node('A', 'Carga CSV')
dot.node('B', 'Limpieza de texto')
dot.node('C', 'Vectorización TF-IDF')
dot.node('D', 'Entrenamiento Naive Bayes')
dot.node('E', 'Guardar modelo')
dot.node('F', 'Clasificación de texto')
dot.node('G', 'Respuesta al usuario')

dot.edges(['AB', 'BC', 'CD', 'DE', 'EF', 'FG'])

dot.render('flujo_entrenamiento', format='png', cleanup=False)
dot

# Interacción en bucle
while True:
    pregunta_usuario = input("Podes preguntarme algo o podes escribir 'salir': ")
    if pregunta_usuario.lower() == "salir":
        break
    print(obtener_respuesta(pregunta_usuario))
