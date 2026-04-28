"""
Modelos de datos del sistema JustIA MVP.

Define las dataclasses que representan las estructuras de datos principales
utilizadas por todos los módulos del Consultorio Jurídico Virtual:
clasificación, NER, QA, clustering, trazabilidad y explicabilidad (XAI).

Cada dataclass incluye validaciones de tipos y valores por defecto según
el diseño del sistema. Estas estructuras garantizan consistencia en la
comunicación entre módulos y facilitan la serialización a JSON para
trazabilidad y auditoría.

Valida: Requisitos 2.1, 3.4, 4.2, 5.2, 7.1
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RegistroCorpus:
    """Registro individual del corpus legal procesado.

    Almacena tanto el texto original como su versión normalizada,
    permitiendo trazabilidad completa del preprocesamiento.

    Attributes:
        id: Identificador único del registro (e.g. "caso_001").
        original: Texto legal en bruto tal como fue recibido.
        limpio: Texto normalizado tras preprocesamiento (minúsculas,
            sin caracteres especiales, lematizado).
        metadata: Información adicional del registro (fuente, fecha, etc.).
    """

    id: str
    original: str
    limpio: str
    metadata: dict = field(default_factory=dict)


@dataclass
class EntidadLegal:
    """Entidad identificada en un texto jurídico por el módulo NER.

    Representa una mención de norma legal, jurisdicción, tipo de violencia
    o persona encontrada en un documento, con su posición exacta en el texto.

    Attributes:
        texto: Texto literal de la entidad encontrada.
        etiqueta: Categoría de la entidad. Valores válidos:
            NORMA_LEGAL | JURISDICCION | VIOLENCIA_TIPO | PERSONA.
        inicio: Índice de inicio de la entidad en el texto original.
        fin: Índice de fin de la entidad en el texto original.
    """

    texto: str
    etiqueta: str  # NORMA_LEGAL | JURISDICCION | VIOLENCIA_TIPO | PERSONA
    inicio: int
    fin: int


@dataclass
class ResultadoClasificacion:
    """Resultado de clasificación temática de un texto legal.

    Contiene la categoría jurídica asignada junto con métricas de
    explicabilidad (XAI) y estado de supervisión humana obligatoria.

    Attributes:
        texto_entrada: Texto legal que fue clasificado.
        categoria: Categoría jurídica asignada. Valores válidos:
            Civil | Penal | Laboral | Familia.
        nivel_certeza: Confianza del modelo en la predicción, rango [0.0, 1.0].
        certeza_porcentaje: Nivel de certeza formateado como porcentaje (e.g. "85.3%").
        key_tokens: Tokens que influyeron en la decisión del modelo,
            ordenados por relevancia descendente.
        requiere_revision: Indica si la predicción necesita revisión humana
            prioritaria (certeza bajo umbral).
        estado_supervision: Estado de validación humana. Valores:
            "Pendiente de Validación Humana" | "Prioridad Alta para Revisión".
        aviso_ia: Aviso obligatorio de que el resultado fue generado por IA.
    """

    texto_entrada: str
    categoria: str  # Civil | Penal | Laboral | Familia
    nivel_certeza: float  # 0.0 a 1.0
    certeza_porcentaje: str  # "85.3%"
    key_tokens: list[str]  # Ordenados por relevancia descendente
    requiere_revision: bool
    estado_supervision: str  # "Pendiente de Validación Humana" | "Prioridad Alta para Revisión"
    aviso_ia: str


@dataclass
class ResultadoNER:
    """Resultado de extracción de entidades nombradas de un texto jurídico.

    Contiene la lista de entidades legales identificadas con sus posiciones,
    junto con estado de supervisión humana obligatoria.

    Attributes:
        texto_entrada: Texto jurídico analizado.
        entidades: Lista de entidades legales encontradas.
        total_entidades: Número total de entidades identificadas.
        mensaje: Mensaje informativo opcional, usado cuando no se encuentran
            entidades para sugerir reformular la consulta.
        estado_supervision: Estado de validación humana. Por defecto
            "Pendiente de Validación Humana".
        aviso_ia: Aviso obligatorio de que el resultado fue generado por IA.
    """

    texto_entrada: str
    entidades: list[EntidadLegal]
    total_entidades: int
    mensaje: Optional[str] = None
    estado_supervision: str = "Pendiente de Validación Humana"
    aviso_ia: str = ""


@dataclass
class ResultadoQA:
    """Resultado de pregunta-respuesta sobre normativa colombiana.

    Contiene la respuesta extraída de la base de conocimiento junto con
    la fuente normativa, métricas de explicabilidad (XAI) y estado de
    supervisión humana obligatoria.

    Attributes:
        pregunta: Pregunta formulada por el usuario consultante.
        respuesta: Respuesta extraída de la base de conocimiento.
        fuente_normativa: Fuente normativa colombiana de donde se extrajo
            la respuesta (e.g. "Ley 1098 de 2006").
        nivel_certeza: Confianza del modelo en la respuesta, rango [0.0, 1.0].
        certeza_porcentaje: Nivel de certeza formateado como porcentaje.
        key_tokens: Tokens relevantes para la respuesta, ordenados por
            relevancia descendente.
        requiere_revision: Indica si la respuesta necesita revisión humana
            prioritaria (certeza bajo umbral).
        estado_supervision: Estado de validación humana.
        aviso_ia: Aviso obligatorio de que el resultado fue generado por IA.
    """

    pregunta: str
    respuesta: str
    fuente_normativa: str
    nivel_certeza: float
    certeza_porcentaje: str
    key_tokens: list[str]
    requiere_revision: bool
    estado_supervision: str
    aviso_ia: str


@dataclass
class InfoCluster:
    """Información de un cluster individual del agrupamiento temático.

    Describe un grupo de casos legales similares con sus palabras clave
    representativas y un resumen descriptivo.

    Attributes:
        cluster_id: Identificador numérico del cluster.
        num_casos: Número de casos asignados a este cluster.
        palabras_clave: Las 3 palabras más representativas del cluster.
        resumen: Resumen descriptivo del tema del cluster.
    """

    cluster_id: int
    num_casos: int
    palabras_clave: list[str]  # Top 3 palabras representativas
    resumen: str


@dataclass
class ResultadoClustering:
    """Resultado de agrupamiento temático de casos legales.

    Contiene los clusters generados con sus métricas de calidad,
    junto con estado de supervisión humana obligatoria.

    Attributes:
        clusters: Lista de clusters con su información detallada.
        num_clusters: Número total de clusters generados.
        num_casos_total: Número total de casos procesados.
        inercia: Métrica de calidad del agrupamiento (suma de distancias
            al centroide). Valores menores indican clusters más compactos.
        estado_supervision: Estado de validación humana. Por defecto
            "Pendiente de Validación Humana".
        aviso_ia: Aviso obligatorio de que el resultado fue generado por IA.
    """

    clusters: list[InfoCluster]
    num_clusters: int
    num_casos_total: int
    inercia: float
    estado_supervision: str = "Pendiente de Validación Humana"
    aviso_ia: str = ""


@dataclass
class RegistroTrazabilidad:
    """Registro individual de trazabilidad para auditoría.

    Documenta una operación ejecutada por cualquier módulo del sistema,
    incluyendo datos de entrada anonimizados y el resultado producido.

    Attributes:
        timestamp: Marca temporal en formato ISO 8601 (e.g. "2024-01-15T10:30:05").
        modulo: Nombre del módulo que ejecutó la operación
            (e.g. "clasificador", "ner", "qa", "clustering").
        version_modelo: Versión o identificador del modelo utilizado.
        entrada_anonimizada: Datos de entrada con información personal
            reemplazada por marcadores genéricos.
        resultado: Diccionario con el resultado de la operación.
        metadata: Información adicional de la operación.
    """

    timestamp: str  # ISO 8601
    modulo: str
    version_modelo: str
    entrada_anonimizada: str
    resultado: dict
    metadata: dict = field(default_factory=dict)


@dataclass
class SesionTrazabilidad:
    """Sesión completa de trazabilidad del sistema.

    Agrupa todos los registros de una ejecución del MVP, incluyendo
    las versiones de modelos y dependencias del entorno para
    garantizar reproducibilidad.

    Attributes:
        inicio_sesion: Marca temporal de inicio en formato ISO 8601.
        versiones_modelos: Diccionario con las versiones de cada modelo
            cargado (e.g. {"clasificador": "mDeBERTa-v3-base-mnli-xnli"}).
        dependencias_entorno: Diccionario con las versiones de las
            dependencias del entorno (e.g. {"python": "3.10", "torch": "2.1.0"}).
        registros: Lista de registros de trazabilidad de la sesión.
    """

    inicio_sesion: str
    versiones_modelos: dict
    dependencias_entorno: dict
    registros: list[RegistroTrazabilidad] = field(default_factory=list)
