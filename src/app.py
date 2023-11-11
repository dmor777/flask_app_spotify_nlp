from pickle import load
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, render_template


model = load(open('nn_6_auto_cosine.model',"rb"))
df = pd.read_excel("datos_merged_1986_2023.xlsx")
df['year_s'] = df['year'].astype(str)
df['duration_ms_s'] = df['duration_ms'].astype(str)
df['popularity_s'] = df['popularity'].astype(str)
df['tags'] = df['track_name'] + " " + df['popularity_s']+ " "+ df['duration_ms_s'] \
    + " " + df['artist_genres']+ " " + df['year_s']
df['tags'] = df['tags'].apply(lambda x: str(x).replace(";"," "))
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['tags'])

def lista_canciones(cancion):
    indice_cancion = df[df['track_name'] == cancion].index[0]
    distancia, indices = model.kneighbors(tfidf_matrix[indice_cancion])
    canciones_similares = [(df['track_name'][i],distancia[0][j]) for j,i in enumerate(indices[0])]
    return canciones_similares[1:]

def str_canciones_recomendadas(cancion_input):
    recomendaciones = lista_canciones(cancion_input)
    resultado = "Recomendaciones para " + cancion_input + "<br />"
    for cancion, distancia in recomendaciones:
        resultado = resultado+"-Canción: "+cancion+"<br />"
    return resultado

app = Flask(__name__)
@app.route("/",methods = ['GET','POST'])
def index():
    if request.method == 'POST':
        nombre_cancion = str(request.form['cancion'])
        lista = str_canciones_recomendadas(nombre_cancion)
    else:
        lista = None
    return render_template("index.html", lista_cancion = lista)