import json
from actividad_1 import preprocesar_texto

def crear_diccionario_base():
    """Crea el archivo JSON inicial si no existe."""
    diccionario = {
        "PENAL": ["delito", "fiscalia", "hurto", "denuncia", "captura"],
        "FAMILIA": ["alimento", "custodia", "divorcio", "conyugal", "menor"],
        "LABORAL": ["contrato", "salario", "despido", "prestacion", "liquidacion"],
        "CONSTITUCIONAL": ["tutela", "peticion", "derecho", "fundamental", "amparo"]
    }
    with open("diccionario.json", "w", encoding="utf-8") as f:
        json.dump(diccionario, f, ensure_ascii=False, indent=4)
    print("[Actividad 2]: Diccionario base creado.")

def clasificar_consulta(texto):
    """Clasifica un texto basado en el diccionario JSON."""
    try:
        with open("diccionario.json", "r", encoding="utf-8") as f:
            categorias = json.load(f)
    except FileNotFoundError:
        crear_diccionario_base()
        return clasificar_consulta(texto)

    texto_limpio = preprocesar_texto(texto)
    puntuaciones = {cat: 0 for cat in categorias.keys()}
    
    for cat, palabras in categorias.items():
        for palabra in palabras:
            if palabra in texto_limpio:
                puntuaciones[cat] += 1
                
    ganador = max(puntuaciones, key=puntuaciones.get)
    return ganador if puntuaciones[ganador] > 0 else "Análisis Humano Requerido"