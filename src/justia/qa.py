"""
Módulo de Preguntas y Respuestas (QA) del sistema JustIA MVP.

Responde preguntas legales frecuentes usando el modelo BERT Spanish-SQuAD
con inferencia directa de tensores PyTorch. Las respuestas se extraen
exclusivamente de la base de conocimiento restringida a normativa
colombiana, garantizando que ninguna respuesta proviene de fuentes
externas al marco jurídico colombiano.

Cada respuesta incluye la fuente normativa específica, key tokens para
explicabilidad (XAI), nivel de certeza, estado de supervisión humana
obligatoria y registro de trazabilidad.

Valida: Requisitos 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8
"""

import logging
import re
from typing import Optional

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from justia.modelos import ResultadoQA
from justia.trazabilidad import ServicioTrazabilidad
from justia.xai import ServicioXAI

logger = logging.getLogger(__name__)


class ModuloQA:
    """Preguntas y Respuestas sobre normativa colombiana.

    Utiliza el modelo BERT Spanish-SQuAD para responder preguntas legales
    frecuentes buscando exclusivamente en la base de conocimiento de
    normativa colombiana. Itera sobre todos los fragmentos normativos
    y selecciona la respuesta con mayor confianza.

    Cada respuesta incluye fuente normativa, key tokens (XAI), nivel de
    certeza, estado de supervisión humana y registro de trazabilidad.

    Attributes:
        MODELO: Identificador del modelo HuggingFace utilizado.
        tokenizer: Tokenizador BERT para el modelo Spanish-SQuAD.
        modelo: Modelo BERT para Question Answering.
        base_conocimiento: Lista de fragmentos normativos colombianos.
        umbral_confianza: Umbral mínimo de certeza para respuestas válidas.
        xai: Servicio de explicabilidad para formatear salidas.
        trazabilidad: Servicio de trazabilidad para registrar operaciones.
    """

    MODELO = "mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"

    def __init__(
        self,
        base_conocimiento: list[dict],
        umbral_confianza: float = 0.5,
        trazabilidad: Optional[ServicioTrazabilidad] = None,
        xai: Optional[ServicioXAI] = None,
    ):
        """Carga el modelo BERT y la base de conocimiento.

        Inicializa el tokenizador y modelo BERT Spanish-SQuAD para
        inferencia directa con tensores PyTorch. La base de conocimiento
        debe contener fragmentos normativos colombianos con campos
        'contexto', 'fuente' y 'tema'.

        Args:
            base_conocimiento: Lista de fragmentos normativos con campos
                'contexto', 'fuente', 'tema'.
            umbral_confianza: Umbral mínimo de certeza para considerar
                una respuesta como válida. Rango [0.0, 1.0].
                Por defecto 0.5.
            trazabilidad: Instancia opcional de ServicioTrazabilidad para
                registrar cada consulta realizada.
            xai: Instancia opcional de ServicioXAI para formatear
                explicaciones y etiquetas de supervisión.
        """
        self.umbral_confianza = umbral_confianza
        self.base_conocimiento = base_conocimiento
        self.trazabilidad = trazabilidad
        self.xai = xai if xai is not None else ServicioXAI()

        # Cargar tokenizador y modelo BERT Spanish-SQuAD
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODELO)
            self.modelo = AutoModelForQuestionAnswering.from_pretrained(
                self.MODELO
            )
            self.modelo.eval()
            logger.info("Modelo QA %s cargado correctamente.", self.MODELO)
        except Exception as e:
            logger.error(
                "Error al cargar el modelo QA %s: %s. "
                "Verifique su conexión a internet o que el modelo "
                "esté disponible localmente.",
                self.MODELO,
                e,
            )
            raise

    def responder(self, pregunta: str) -> ResultadoQA:
        """Busca respuesta en la base de conocimiento colombiana.

        Itera sobre todos los fragmentos normativos de la base de
        conocimiento, ejecutando inferencia con tensores PyTorch para
        cada uno. Selecciona la respuesta con mayor confianza y la
        retorna junto con la fuente normativa, key tokens (XAI),
        nivel de certeza y estado de supervisión humana.

        Si la mejor certeza es inferior al umbral de confianza, se
        informa que la pregunta requiere atención del Supervisor Humano.

        Args:
            pregunta: Pregunta del usuario consultante.

        Returns:
            ResultadoQA con respuesta, fuente_normativa, nivel_certeza,
            certeza_porcentaje, key_tokens, requiere_revision,
            estado_supervision y aviso_ia.
        """
        mejor_respuesta = ""
        mejor_certeza = 0.0
        mejor_fuente = ""
        mejor_contexto = ""

        # Iterar sobre todos los fragmentos de la base de conocimiento
        for fragmento in self.base_conocimiento:
            contexto = fragmento.get("contexto", "")
            fuente = fragmento.get("fuente", "")

            if not contexto:
                continue

            # Inferencia directa con tensores PyTorch
            respuesta_texto, score = self._inferencia_tensores(
                pregunta, contexto
            )

            # Conservar la mejor respuesta (mayor confianza)
            if score > mejor_certeza and respuesta_texto:
                mejor_certeza = score
                mejor_respuesta = respuesta_texto
                mejor_fuente = fuente
                mejor_contexto = contexto

        # Si certeza < umbral, informar que requiere Supervisor Humano
        if mejor_certeza < self.umbral_confianza:
            mejor_respuesta = (
                "La pregunta requiere atención del Supervisor Humano. "
                "No se encontró una respuesta con suficiente certeza "
                "en la base de conocimiento."
            )

        # Extraer key tokens para explicabilidad (XAI)
        key_tokens = self._extraer_key_tokens(
            pregunta, mejor_contexto, mejor_respuesta
        )

        # Formatear explicación XAI
        explicacion = self.xai.formatear_explicacion(
            key_tokens=key_tokens,
            nivel_certeza=mejor_certeza,
            umbral_confianza=self.umbral_confianza,
        )

        # Generar etiquetas de supervisión humana
        supervision = self.xai.etiquetar_supervision(
            nivel_certeza=mejor_certeza,
            umbral_confianza=self.umbral_confianza,
        )

        # Construir resultado QA
        resultado = ResultadoQA(
            pregunta=pregunta,
            respuesta=mejor_respuesta,
            fuente_normativa=mejor_fuente,
            nivel_certeza=mejor_certeza,
            certeza_porcentaje=explicacion["certeza_porcentaje"],
            key_tokens=explicacion["key_tokens"],
            requiere_revision=explicacion["requiere_revision"],
            estado_supervision=supervision["estado"],
            aviso_ia=supervision["aviso"],
        )

        # Registrar en trazabilidad
        if self.trazabilidad is not None:
            self.trazabilidad.registrar_operacion(
                modulo="qa",
                entrada=pregunta,
                resultado={
                    "respuesta": mejor_respuesta,
                    "fuente_normativa": mejor_fuente,
                    "nivel_certeza": mejor_certeza,
                    "key_tokens": key_tokens,
                },
                version_modelo=self.MODELO,
                metadata={
                    "umbral_confianza": self.umbral_confianza,
                    "requiere_revision": explicacion["requiere_revision"],
                },
            )

        return resultado

    def _inferencia_tensores(
        self, pregunta: str, contexto: str
    ) -> tuple[str, float]:
        """Realiza inferencia directa con tensores PyTorch.

        Tokeniza la pregunta y el contexto, ejecuta el modelo BERT
        para obtener logits de inicio y fin, y extrae el span de
        respuesta del contexto. La confianza se calcula usando softmax
        sobre los logits.

        Args:
            pregunta: Pregunta del usuario consultante.
            contexto: Fragmento normativo donde buscar la respuesta.

        Returns:
            Tupla (respuesta_texto, score_confianza) donde
            respuesta_texto es el span extraído del contexto y
            score_confianza es la confianza del modelo en [0.0, 1.0].
        """
        # Tokenizar pregunta + contexto
        inputs = self.tokenizer(
            pregunta,
            contexto,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        # Inferencia sin gradientes (modo evaluación)
        with torch.no_grad():
            outputs = self.modelo(**inputs)

        # Obtener logits de inicio y fin
        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]

        # Calcular probabilidades con softmax
        start_probs = torch.softmax(start_logits, dim=0)
        end_probs = torch.softmax(end_logits, dim=0)

        # Obtener posiciones de inicio y fin con mayor probabilidad
        start_idx = torch.argmax(start_probs).item()
        end_idx = torch.argmax(end_probs).item()

        # Asegurar que end >= start; si no, la respuesta no es válida
        if end_idx < start_idx:
            return "", 0.0

        # Calcular score de confianza como producto de probabilidades
        score = (start_probs[start_idx] * end_probs[end_idx]).item()

        # Extraer tokens de la respuesta y decodificar
        input_ids = inputs["input_ids"][0]
        answer_ids = input_ids[start_idx : end_idx + 1]
        respuesta_texto = self.tokenizer.decode(
            answer_ids, skip_special_tokens=True
        ).strip()

        # Si la respuesta es un token especial o vacía, retornar vacío
        if not respuesta_texto or respuesta_texto in ("[CLS]", "[SEP]"):
            return "", 0.0

        return respuesta_texto, score

    def _extraer_key_tokens(
        self, pregunta: str, contexto: str, respuesta: str
    ) -> list[str]:
        """Extrae tokens relevantes para la respuesta (XAI).

        Identifica las palabras más relevantes combinando tokens de la
        respuesta con palabras del contexto que aparecen en la pregunta.
        Los tokens se ordenan por relevancia: primero los de la respuesta,
        luego los del contexto que coinciden con la pregunta.

        Args:
            pregunta: Pregunta del usuario consultante.
            contexto: Fragmento normativo utilizado como contexto.
            respuesta: Respuesta extraída del contexto.

        Returns:
            Lista de tokens relevantes ordenados por relevancia
            descendente. Máximo 10 tokens.
        """
        # Extraer palabras significativas (más de 2 caracteres)
        def extraer_palabras(texto: str) -> list[str]:
            return [
                p
                for p in re.findall(r"[a-záéíóúñü]+", texto.lower())
                if len(p) > 2
            ]

        palabras_respuesta = extraer_palabras(respuesta)
        palabras_pregunta = set(extraer_palabras(pregunta))
        palabras_contexto = extraer_palabras(contexto)

        # Tokens de la respuesta (máxima relevancia)
        tokens_respuesta: list[str] = []
        vistos: set[str] = set()
        for palabra in palabras_respuesta:
            if palabra not in vistos:
                tokens_respuesta.append(palabra)
                vistos.add(palabra)

        # Tokens del contexto que coinciden con la pregunta
        tokens_contexto: list[str] = []
        for palabra in palabras_contexto:
            if palabra in palabras_pregunta and palabra not in vistos:
                tokens_contexto.append(palabra)
                vistos.add(palabra)

        # Combinar: primero respuesta, luego contexto relevante
        key_tokens = tokens_respuesta + tokens_contexto

        # Si no hay tokens suficientes, agregar palabras largas del contexto
        if len(key_tokens) < 3:
            palabras_largas = sorted(
                [p for p in set(palabras_contexto) if p not in vistos],
                key=len,
                reverse=True,
            )
            key_tokens.extend(palabras_largas[: 3 - len(key_tokens)])

        # Limitar a máximo 10 tokens
        return key_tokens[:10]
