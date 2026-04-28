"""
Preprocesador de textos legales del sistema JustIA MVP.

Normaliza textos legales colombianos para consumo por los módulos de IA,
aplicando conversión a minúsculas, eliminación de caracteres especiales
y lematización mediante spaCy. Procesa corpus completos excluyendo
registros inválidos y registrando advertencias en trazabilidad.

Valida: Requisitos 1.1, 1.2, 1.3, 1.4
"""

import re
from typing import Optional

import spacy

from justia.trazabilidad import ServicioTrazabilidad


class Preprocesador:
    """Normaliza textos legales colombianos para los módulos de IA.

    Aplica un pipeline de normalización que incluye conversión a minúsculas,
    eliminación de caracteres especiales (conservando solo alfanuméricos y
    espacios) y lematización con spaCy. Integra opcionalmente con el
    ServicioTrazabilidad para registrar advertencias de textos inválidos.

    Attributes:
        nlp: Modelo spaCy cargado para lematización.
        trazabilidad: Servicio de trazabilidad opcional para registrar
            advertencias de textos inválidos.
    """

    # Patrón regex: conservar solo caracteres alfanuméricos y espacios
    _PATRON_ESPECIALES = re.compile(r"[^a-záéíóúñü0-9\s]", re.IGNORECASE)

    def __init__(
        self,
        nlp_model: str = "es_core_news_sm",
        trazabilidad: Optional[ServicioTrazabilidad] = None,
    ):
        """Carga el modelo spaCy para lematización.

        Args:
            nlp_model: Nombre del modelo spaCy a cargar. Por defecto
                ``es_core_news_sm`` (español).
            trazabilidad: Instancia opcional de ServicioTrazabilidad para
                registrar advertencias cuando se encuentran textos inválidos
                durante el procesamiento del corpus.

        Raises:
            OSError: Si el modelo spaCy no está instalado.
        """
        self.nlp = spacy.load(nlp_model)
        self.trazabilidad = trazabilidad

    def normalizar_texto(self, texto: str) -> str:
        """Aplica minúsculas, eliminación de caracteres especiales y lematización.

        Pipeline de normalización:
        1. Conversión a minúsculas.
        2. Eliminación de caracteres especiales (conserva solo alfanuméricos
           y espacios).
        3. Lematización con spaCy.

        Args:
            texto: Texto legal en bruto.

        Returns:
            Texto normalizado y lematizado.

        Raises:
            ValueError: Si el texto está vacío o solo contiene caracteres
                especiales (sin contenido alfanumérico útil).
        """
        # 1. Conversión a minúsculas
        texto_lower = texto.lower()

        # 2. Eliminación de caracteres especiales
        texto_limpio = self._PATRON_ESPECIALES.sub("", texto_lower)

        # Colapsar espacios múltiples y recortar
        texto_limpio = re.sub(r"\s+", " ", texto_limpio).strip()

        # Validar que quede contenido alfanumérico útil
        if not texto_limpio or not any(c.isalnum() for c in texto_limpio):
            raise ValueError(
                "El texto está vacío o solo contiene caracteres especiales."
            )

        # 3. Lematización con spaCy
        doc = self.nlp(texto_limpio)
        lemas = [token.lemma_ for token in doc if not token.is_space]
        texto_lematizado = " ".join(lemas)

        return texto_lematizado

    def procesar_corpus(self, corpus_crudo: list[dict]) -> list[dict]:
        """Procesa un corpus completo, retornando registros con campos 'original' y 'limpio'.

        Itera sobre cada registro del corpus crudo, normaliza el texto y
        construye un diccionario con el texto original y su versión limpia.
        Los registros con textos inválidos se excluyen y se registra una
        advertencia en el servicio de trazabilidad.

        Args:
            corpus_crudo: Lista de diccionarios con campo ``'texto'``.

        Returns:
            Lista de diccionarios con campos ``'original'`` (texto sin
            modificar) y ``'limpio'`` (texto normalizado y lematizado).
            Los registros inválidos se excluyen del resultado.
        """
        corpus_procesado = []

        for registro in corpus_crudo:
            texto_original = registro.get("texto", "")

            try:
                texto_limpio = self.normalizar_texto(texto_original)
                corpus_procesado.append({
                    "original": texto_original,
                    "limpio": texto_limpio,
                })
            except ValueError as e:
                # Registrar advertencia en trazabilidad si está disponible
                if self.trazabilidad is not None:
                    self.trazabilidad.registrar_operacion(
                        modulo="preprocesador",
                        entrada=texto_original,
                        resultado={
                            "estado": "excluido",
                            "motivo": str(e),
                        },
                        version_modelo="es_core_news_sm",
                        metadata={"tipo": "advertencia"},
                    )

        return corpus_procesado
