import pandas as pd
import re
import spacy

# Carga global del modelo para evitar recargas lentas
try:
    nlp = spacy.load("es_core_news_sm")
except:
    import os
    os.system("python -m spacy download es_core_news_sm")
    nlp = spacy.load("es_core_news_sm")

def preprocesar_texto(texto):
    """Limpia, tokeniza y lematiza el texto jurídico."""
    # Limpieza básica
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = re.sub(r'\d+', '', texto)
    
    # Procesamiento NLU
    doc = nlp(texto)
    # Filtrado de stop words y palabras cortas
    tokens = [token.lemma_ for token in doc if not token.is_stop and len(token.text) > 3]
    return " ".join(tokens)

def ejecutar_actividad_1():
    print("[Actividad 1]: Iniciando preprocesamiento del corpus...")
    corpus_original = [
        "Demanda por alimentos para menor de edad.",
        "Acción de tutela por derecho a la salud.",
        "Despido injustificado y liquidación de contrato.",
        "Hurto agravado en establecimiento comercial."
    ] * 13 # Simula 52 registros
    
    df = pd.DataFrame(corpus_original, columns=['original'])
    df['limpio'] = df['original'].apply(preprocesar_texto)
    
    # Guardar en la carpeta actual
    df.to_json("corpus_justia_limpio.json", orient="records", force_ascii=False, indent=4)
    print("[Actividad 1]: Corpus guardado en 'corpus_justia_limpio.json'.")
    return df