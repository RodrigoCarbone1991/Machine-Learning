import pandas as pd
import unicodedata
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

def limpiar_texto(texto):
    texto = texto.lower()
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto)
                    if unicodedata.category(c) != 'Mn')
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

dataset = pd.read_csv("datos/data_set.csv", skipinitialspace=True)

dataset['Pregunta_limpia'] = dataset['Pregunta'].apply(limpiar_texto)

preguntas = dataset['Pregunta_limpia'].values
respuestas = dataset['Respuesta'].values

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(preguntas)

modelo = MultinomialNB()
modelo.fit(X, respuestas)

joblib.dump(modelo, "modelo_entrenado.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

def obtener_respuesta(pregunta_usuario):
    pregunta_limpia = limpiar_texto(pregunta_usuario)
    pregunta_vectorizada = vectorizer.transform([pregunta_limpia])
    probas = modelo.predict_proba(pregunta_vectorizada)[0]
    indice_max = probas.argmax()
    confianza = probas[indice_max]

    if confianza < 0.3:
     return "No entendi lo que me preguntaste, me lo podes repetir?"

    respuesta = modelo.classes_[indice_max]
    return respuesta

while True:
    pregunta_usuario = input("Podes preguntarme algo o podes escribir 'salir': ")
    if pregunta_usuario.lower() == "salir":
        break
    print(obtener_respuesta(pregunta_usuario))
