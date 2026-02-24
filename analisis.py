import os
import pandas as pd
import spacy
from pysentimiento import create_analyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import gdown  # Necesaria para descargar de Drive

# --- CONFIGURACIÓN DE ENTORNO ---
os.environ["PYSENTIMIENTO_BACKEND"] = "torch"

# --- 1. DESCARGA DE EXCEL DESDE GOOGLE DRIVE ---
# Reemplaza 'TU_ID_AQUÍ' con el ID que obtuviste de tu enlace compartido
FILE_ID = '1yEetHbQFF8Y8EbxlDRibDH8FwtOl8V27' 
url = f'https://drive.google.com/uc?id={FILE_ID}'
archivo_destino = 'reseñas_descargadas.xlsx'

print("Descargando archivo desde Google Drive...")
gdown.download(url, archivo_destino, quiet=False)

# --- 2. CARGA Y LIMPIEZA ---
df = pd.read_excel(archivo_destino)

df["comentario"] = df["comentario"].astype(str).str.lower()
df['comentario'] = df['comentario'].str.replace(r'\n', ' ', regex=True)
df['comentario'] = df['comentario'].str.replace(r'\s+', ' ', regex=True).str.strip()

# --- 3. INICIALIZACIÓN DE MODELOS ---
print("Cargando modelos de IA (esto puede tardar en la nube)...")
analizador = create_analyzer(task="sentiment", lang="es")
# Para la nube usamos el modelo pequeño para que sea más rápido
try:
    nlp = spacy.load("es_core_news_sm", disable=["ner", "parser"])
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "es_core_news_sm"])
    nlp = spacy.load("es_core_news_sm", disable=["ner", "parser"])

# --- 4. PROCESAMIENTO ---
sentimientos, prob_pos, prob_neu, prob_neg, keywords = [], [], [], [], []

print(f"Iniciando análisis de {len(df)} reseñas...")

for i in range(len(df)):
    texto = df['comentario'].iloc[i]
    
    # Sentimiento
    res = analizador.predict(texto)
    sentimientos.append(res.output)
    prob_pos.append(round(res.probas['POS'], 2))
    prob_neu.append(round(res.probas['NEU'], 2))
    prob_neg.append(round(res.probas['NEG'], 2))
    
    # Keywords con spaCy
    doc = nlp(texto)
    lemas = [t.lemma_ for t in doc if t.pos_ in ["NOUN", "ADJ"] and not t.is_stop]
    keywords.append(", ".join(lemas))

df['Sentimiento_Final'] = sentimientos
df['Prob_Positivo'] = prob_pos
df['Prob_Negativo'] = prob_neg
df['Analisis_Keywords'] = keywords

# --- 5. GENERACIÓN DE REPORTES Y GRÁFICAS ---
conteo_sentimientos = df["Sentimiento_Final"].value_counts()
print("\n--- REPORTE FINAL ---")
print(conteo_sentimientos)

# Crear Gráfica de Barras
plt.figure(figsize=(10, 6))
conteo_sentimientos.plot(kind='bar', color=['green', 'gray', 'red'])
plt.title('Análisis de Sentimientos Docentes')
plt.xlabel('Sentimiento')
plt.ylabel('Cantidad de Reseñas')

# GUARDAR RESULTADO (Vital para GitHub Actions)
plt.savefig("reporte_final_sentimientos.png")
print("\nGráfica guardada como 'reporte_final_sentimientos.png'")

# Guardar Excel procesado
df.to_excel("analisis_resultados_completo.xlsx", index=False)
print("Excel de resultados generado.")