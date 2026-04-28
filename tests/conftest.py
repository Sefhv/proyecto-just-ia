"""
Fixtures compartidos y generadores Hypothesis reutilizables para JustIA MVP.

Este módulo centraliza:
- Estrategias Hypothesis para generar datos de prueba (textos legales,
  entidades, niveles de confianza, patrones de leyes colombianas, etc.)
- Fixtures pytest compartidos entre todos los archivos de test.
"""

import pytest
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Estrategias Hypothesis reutilizables
# ---------------------------------------------------------------------------

# Categorías jurídicas válidas del sistema
CATEGORIAS_JURIDICAS = ["Civil", "Penal", "Laboral", "Familia"]

# Etiquetas NER válidas del sistema
ETIQUETAS_NER = ["NORMA_LEGAL", "JURISDICCION", "VIOLENCIA_TIPO", "PERSONA"]

# Aviso de IA obligatorio (Requisito 8.2)
AVISO_IA = (
    "Este resultado fue generado por IA y requiere validación de un "
    "profesional del derecho antes de ser utilizado."
)


def textos_legales_st(min_size: int = 10, max_size: int = 300) -> st.SearchStrategy:
    """Genera textos que simulan fragmentos legales con caracteres alfanuméricos y espacios.

    Útil para tests de clasificación, preprocesamiento y NER donde se necesita
    texto con al menos algunos caracteres alfanuméricos.
    """
    return st.text(
        min_size=min_size,
        max_size=max_size,
        alphabet=st.characters(whitelist_categories=("L", "N", "Z")),
    ).filter(lambda t: any(c.isalnum() for c in t))


def textos_invalidos_st() -> st.SearchStrategy:
    """Genera textos inválidos: vacíos o compuestos solo de caracteres especiales/espacios.

    Útil para tests de rechazo de entradas inválidas (Propiedad 3).
    """
    return st.one_of(
        st.just(""),
        st.text(
            alphabet=st.characters(whitelist_categories=("P", "Z", "S")),
            min_size=0,
            max_size=50,
        ),
    )


def numeros_ley_st() -> st.SearchStrategy:
    """Genera números de ley colombiana (entero positivo entre 1 y 2500)."""
    return st.integers(min_value=1, max_value=2500)


def anios_ley_st() -> st.SearchStrategy:
    """Genera años válidos para leyes colombianas (1887–2024).

    1887 es el año de la primera Constitución de Colombia.
    """
    return st.integers(min_value=1887, max_value=2024)


def patrones_ley_colombiana_st() -> st.SearchStrategy:
    """Genera textos con formato 'Ley [número] de [año]' embebidos en contexto.

    Útil para tests de reconocimiento de normas legales (Propiedad 8).
    """
    return st.builds(
        lambda num, anio: f"Según la Ley {num} de {anio}, se establece que",
        num=numeros_ley_st(),
        anio=anios_ley_st(),
    )


def niveles_certeza_st() -> st.SearchStrategy:
    """Genera niveles de certeza válidos en el rango [0.0, 1.0].

    Útil para tests de XAI, supervisión y escalamiento (Propiedades 5, 6, 16).
    """
    return st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


def umbrales_confianza_st() -> st.SearchStrategy:
    """Genera umbrales de confianza típicos en el rango [0.3, 0.9]."""
    return st.floats(min_value=0.3, max_value=0.9, allow_nan=False, allow_infinity=False)


def key_tokens_st(min_size: int = 1, max_size: int = 10) -> st.SearchStrategy:
    """Genera listas de key tokens (palabras clave) no vacías.

    Útil para tests de explicabilidad XAI (Propiedad 5).
    """
    token = st.text(
        min_size=2,
        max_size=20,
        alphabet=st.characters(whitelist_categories=("L",)),
    ).filter(lambda t: len(t.strip()) >= 2)
    return st.lists(token, min_size=min_size, max_size=max_size)


def nombres_persona_st() -> st.SearchStrategy:
    """Genera nombres propios colombianos típicos para tests de anonimización.

    Útil para tests de la Propiedad 17 (efectividad de anonimización).
    """
    nombres = [
        "Juan Carlos Pérez",
        "María Fernanda López",
        "Carlos Andrés García",
        "Ana María Rodríguez",
        "Luis Eduardo Martínez",
        "Sandra Patricia Gómez",
        "Jorge Enrique Díaz",
        "Diana Carolina Morales",
    ]
    return st.sampled_from(nombres)


def cedulas_st() -> st.SearchStrategy:
    """Genera números de cédula colombiana (formato: 8-10 dígitos).

    Útil para tests de anonimización (Propiedad 17).
    """
    return st.from_regex(r"[0-9]{8,10}", fullmatch=True)


def telefonos_st() -> st.SearchStrategy:
    """Genera números de teléfono colombiano (formato: 3XX-XXX-XXXX).

    Útil para tests de anonimización (Propiedad 17).
    """
    return st.from_regex(r"3[0-9]{2}[0-9]{3}[0-9]{4}", fullmatch=True)


def textos_con_datos_personales_st() -> st.SearchStrategy:
    """Genera textos que contienen datos personales embebidos.

    Combina nombres, cédulas y teléfonos en contexto legal para tests
    de anonimización (Propiedad 17).
    """
    return st.builds(
        lambda nombre, cedula, tel: (
            f"El demandante {nombre}, identificado con cédula {cedula}, "
            f"con teléfono {tel}, solicita protección legal."
        ),
        nombre=nombres_persona_st(),
        cedula=cedulas_st(),
        tel=telefonos_st(),
    )


def nombres_modulo_st() -> st.SearchStrategy:
    """Genera nombres de módulos válidos del sistema JustIA."""
    return st.sampled_from([
        "preprocesador",
        "clasificador",
        "ner",
        "qa",
        "clustering",
    ])


def versiones_modelo_st() -> st.SearchStrategy:
    """Genera versiones de modelo representativas del sistema."""
    return st.sampled_from([
        "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        "es_core_news_sm",
        "mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es",
        "sklearn TF-IDF + KMeans",
        "es_core_news_sm + EntityRuler v1.0",
    ])


def resultados_operacion_st() -> st.SearchStrategy:
    """Genera diccionarios de resultado para registros de trazabilidad."""
    return st.fixed_dictionaries({
        "estado": st.sampled_from(["exitoso", "error_validacion", "error_inesperado"]),
        "detalle": st.text(min_size=1, max_size=50),
    })


def textos_corpus_st(min_size: int = 50, max_size: int = 60) -> st.SearchStrategy:
    """Genera listas de textos para tests de clustering (mínimo 50 elementos).

    Útil para tests de las Propiedades 11, 12, 13.
    """
    texto = st.text(
        min_size=20,
        max_size=100,
        alphabet=st.characters(whitelist_categories=("L", "N", "Z")),
    ).filter(lambda t: any(c.isalnum() for c in t))
    return st.lists(texto, min_size=min_size, max_size=max_size)


def jurisdicciones_colombianas_st() -> st.SearchStrategy:
    """Genera nombres de jurisdicciones colombianas para tests de NER."""
    return st.sampled_from([
        "Corte Constitucional",
        "Corte Suprema de Justicia",
        "Consejo de Estado",
        "Tribunal Superior de Bogotá",
        "Juzgado Primero Civil",
        "Juzgado de Familia",
        "Fiscalía General de la Nación",
    ])


def tipos_violencia_st() -> st.SearchStrategy:
    """Genera tipos de violencia reconocidos por el sistema NER."""
    return st.sampled_from([
        "violencia intrafamiliar",
        "violencia patrimonial",
        "violencia económica",
    ])


# ---------------------------------------------------------------------------
# Fixtures pytest compartidos
# ---------------------------------------------------------------------------

@pytest.fixture
def categorias_juridicas():
    """Conjunto de categorías jurídicas válidas del sistema."""
    return set(CATEGORIAS_JURIDICAS)


@pytest.fixture
def etiquetas_ner():
    """Conjunto de etiquetas NER válidas del sistema."""
    return set(ETIQUETAS_NER)


@pytest.fixture
def aviso_ia():
    """Texto del aviso de IA obligatorio (Requisito 8.2)."""
    return AVISO_IA


@pytest.fixture
def umbral_confianza_default():
    """Umbral de confianza por defecto para los módulos."""
    return 0.6


@pytest.fixture
def textos_ejemplo():
    """Textos legales de ejemplo para tests unitarios."""
    return [
        "La demandante solicita la disolución del vínculo matrimonial conforme al artículo 154 del Código Civil.",
        "El imputado fue capturado en flagrancia por el delito de hurto calificado según el artículo 240 del Código Penal.",
        "El trabajador reclama el pago de prestaciones sociales adeudadas por su empleador desde hace dos años.",
        "Se solicita la custodia del menor ante el Juzgado de Familia por presunta violencia intrafamiliar.",
    ]


@pytest.fixture
def corpus_minimo():
    """Corpus mínimo de 50 registros para tests de preprocesamiento y clustering."""
    textos_base = [
        "Demanda de divorcio por causal de abandono del hogar conyugal.",
        "Denuncia penal por lesiones personales en riña callejera.",
        "Reclamación laboral por despido sin justa causa del trabajador.",
        "Solicitud de medida de protección por violencia intrafamiliar.",
        "Proceso de sucesión intestada de bienes inmuebles.",
        "Acción de tutela por vulneración del derecho a la salud.",
        "Demanda ejecutiva por cobro de pagaré vencido.",
        "Querella por perturbación de la posesión de predio rural.",
        "Solicitud de conciliación por alimentos para menor de edad.",
        "Denuncia por estafa en contrato de compraventa de vehículo.",
    ]
    corpus = []
    for i in range(50):
        texto = textos_base[i % len(textos_base)]
        corpus.append({
            "id": f"caso_{i+1:03d}",
            "texto": texto,
            "area_esperada": CATEGORIAS_JURIDICAS[i % len(CATEGORIAS_JURIDICAS)],
            "metadata": {"fuente": "sintético", "indice": i},
        })
    return corpus
