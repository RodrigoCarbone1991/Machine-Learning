import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import re
import unicodedata
import snowballstemmer

stemmer = snowballstemmer.stemmer('spanish')

def limpiar_texto(texto):
    texto = texto.lower()
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto)
                    if unicodedata.category(c) != 'Mn')
    texto = re.sub(r'[^\w\s]', '', texto)
    tokens = texto.split()
    texto_stem = ' '.join([stemmer.stemWord(token) for token in tokens])
    return texto_stem


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
    
    rango_confianza = 0.3
    
    ancho = 90
    linea_superior = "╔" + "═" * (ancho - 2) + "╗"
    linea_inferior = "╚" + "═" * (ancho - 2) + "╝"
    linea_separadora = "╠" + "═" * (ancho - 2) + "╣"
    linea_divisoria = "║" + "─" * (ancho - 2) + "║"

    cabecera = "ANALISIS DEL MODELO SEGUN PROMPT"
    items = [
        f"Cantidad de pruebas en el dataset: {len(preguntas)}",
        f"Cantidad de palabras en el vocabulario: {len(vectorizer.get_feature_names_out())}",
        f"Probabilidad de cada clase: {probas}",
        f"Clase con mayor probabilidad: {confianza}",
        f"Nivel de confianza: {rango_confianza}",
        
        f"Categoría elegida: {modelo.classes_[indice_max]}",
    ]

    print(linea_superior)
    print("║" + cabecera.center(ancho - 2) + "║")
    print(linea_separadora)
    for i, item in enumerate(items):
        print("║ " + item.ljust(ancho - 3) + "║")
        if i < len(items) - 1:
            print(linea_divisoria)
    print(linea_inferior)

    if confianza < rango_confianza:
        with open("log_desconocidas.txt", "a") as f:
            f.write(pregunta_usuario + "\n")
        return "No entendi lo que me preguntaste, me lo podes repetir?"
    return modelo.classes_[indice_max]

while True:
    pregunta_usuario = input("Podes preguntarme algo o podes escribir 'salir': ")
    if pregunta_usuario.lower() == "salir":
        break
    print(obtener_respuesta(pregunta_usuario))
