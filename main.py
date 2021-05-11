import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


siteHeader = st.beta_container()
# Será el contenedor del título
with siteHeader: 
    st.title("Clasificador de SPAM")
    st.markdown(""" El objetivo de este proyecto es proveer al usuario una herramienta que permita predecir 
    sí un correo debería ser clasificado como **spam** evaluando una muestra del mensaje. """)

# Empezamos con el modelo
newFeautures = st.beta_container()
with newFeautures: 
    st.header(" Base Inicial")
    st.markdown(""" Demos un vistazo al dataset: """)

# Cargamos la base y damos un preview de la info
data = pd.read_csv('https://raw.githubusercontent.com/ReynaldoMR3/clases_ih/main/emails.csv')
# Marcamos con 1 sí el correo es spam y  si no lo es
data['is_spam'] = data['class'].apply(lambda x: 1 if x == 'spam' else 0)
st.write(data.sample(3))
st.markdown(""" Construiremos nuestro modelo con base en la columna **is_spam**. """)

# Empezamos con el modelo
newFeautures = st.beta_container()
with newFeautures: 
    st.header("Modelo - Naive Bayes")
    st.markdown(""" * Para empezar, vectorizamos la columna **message** aplicando *CountVectorizer()*.""")

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['message'])
y = data['is_spam']

# Creamos nuestros grupos de entrenamiento
st.markdown(""" * Con ayuda de *train_test_split()* entrenamos nuestro modelo clasificador *MultinomialNB()*.""")
st.text(""" Por default se considera el 80% del dataset para el entrenamiento, 
puedes modificar el porcentaje a usar con el siguiente slider:""")
train = st.slider('Tamaño (%) del train.', min_value=60, max_value=90, step=5, value=80)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = (100-train)/100,random_state = 666)

# Entrenamos y predecimos
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
y_predict=classifier.predict(X_test)

# Para evaluar el modelo podemos correr la matriz de confusión
st.markdown(""" * Evaluamos el desempeño del modelo con una matriz de confusión obteniendo los siguientes resultados:""")

matriz = pd.DataFrame(confusion_matrix(y_test, y_predict))
matriz_confusion=pd.DataFrame([matriz[0][0], matriz[1][0], matriz[0][1], matriz[1][1]],['Verdaderos Positivos', 'Falsos Positivos', 'Falsos Negativos','Verdaderos Negativos'])

falla = ((matriz[1][0] + matriz[0][1])/512).round(2)*100

st.bar_chart(matriz_confusion)
st.markdown(""" Como podemos observar, el modelo sólo falló en menos del """ + str(falla) +"""%.""")

# Probando el modelo
otherFeatures = st.beta_container()
with otherFeatures: 
    st.header("Casos de Uso")
    st.markdown(""" Para mostrar el desempeño del modelo, realicemos pruebas con los siguientes ejemplos:""")
    st.markdown("""
    1. You can SAVE hundreds, reply and get a free coupon
    2. Hi, how about a game of poker tomorrow night?
    3. Free coupon inside
    4. I have been expecting you Mr. Anderson""")

# Veamos como se comporta el modelo con algunos mensajes de muestra:
examples = ['You can SAVE hundreds, reply and get a free coupon',
            'Hi, how about a game of poker tomorrow night?', 
            'Free coupon inside', 
            "I've been expecting you Mr. Anderson" ]
            
example_counts = vectorizer.transform(examples)
predictions = classifier.predict(example_counts)

resultados = pd.DataFrame([examples,predictions], index=['Mensaje','is_spam']).T
resultados

# Interacción con el usuario
finalFeatures = st.beta_container()
with finalFeatures: 
    st.header("¡Prueba el modelo! (:")

prueba = st.text_input('Ingresa un mensaje para clasificar:','Auto Loans, Fast Approvals for Any Credit!')
example = vectorizer.transform(['Auto Loans, Fast Approvals for Any Credit!',prueba])

# Resultados
predictions = classifier.predict(example)

# Podemos ver las probabilidades que generó el modelo para cada movimiento
examples_predict = classifier.predict_proba(example)

if predictions[1]==1:
    resultado_prueba = 'Es SPAM'
    proba = examples_predict[1][1].round(3)*100
else: 
    resultado_prueba = 'No es SPAM'
    proba = examples_predict[1][0].round(3)*100

# Output para el usuario    
st.text('Resultado ...')
st.write(resultado_prueba + " con " + str(proba) + "% de probabilidad.")

# Estilo de lo que vamos a ejecutar lo que teniamos allá arriba:
st.markdown( """ <style>
 .main {
 background-color: #AF9EC;
}
</style>""", unsafe_allow_html=True )