"""
Módulo Clasificador de textos legales del sistema JustIA MVP.

Clasifica textos legales colombianos en exactamente una de cuatro
categorías jurídicas (Civil, Penal, Laboral, Familia) mediante
clasificación zero-shot con el modelo mDeBERTa-v3. Integra
explicabilidad (XAI) con key tokens, supervisión humana obligatoria
y trazabilidad de cada operación.

Valida: Requisitos 2.1, 2.2, 2.3, 2.4, 2.5, 2.6
"""

import logging
import re
from typing import Optional

from justia.modelos import ResultadoClasificacion
from justia.trazabilidad import ServicioTrazabilidad
from justia.xai import ServicioXAI

logger = logging.getLogger(__name__)


class ModuloClasificador:
    """Clasificación zero-shot de textos legales colombianos.

    Utiliza el modelo mDeBERTa-v3-base-mnli-xnli para clasificar textos
    legales en una de cuatro categorías jurídicas sin entrenamiento previo
    específico. Cada clasificación incluye key tokens para explicabilidad,
    nivel de certeza, estado de supervisión humana y registro de trazabilidad.

    Attributes:
        CATEGORIAS: Las cuatro categorías jurídicas válidas.
        MODELO: Identificador del modelo HuggingFace utilizado.
        pipeline: Pipeline de zero-shot classification de transformers.
        umbral_confianza: Umbral mínimo de certeza para considerar una
            predicción como confiable (por defecto 0.6).
        xai: Servicio de explicabilidad para formatear salidas.
        trazabilidad: Servicio de trazabilidad para registrar operaciones.
    """

    CATEGORIAS = ["Civil", "Penal", "Laboral", "Familia"]
    MODELO = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

    # Límite de tokens del modelo (basado en palabras como aproximación)
    _MAX_TOKENS = 512

    # Palabras clave asociadas a cada categoría jurídica colombiana,
    # utilizadas como heurística para extraer key tokens relevantes (XAI)
    _KEYWORDS_POR_CATEGORIA = {
        "Civil": {
            "contrato", "propiedad", "arrendamiento", "hipoteca", "sucesión",
            "herencia", "obligación", "demanda", "indemnización", "daño",
            "responsabilidad", "patrimonio", "escritura", "inmueble",
            "compraventa", "deuda", "acreedor", "deudor", "civil",
            "prescripción", "servidumbre", "usufructo", "posesión",
        },
        "Penal": {
            "delito", "pena", "homicidio", "hurto", "robo", "estafa",
            "lesiones", "imputado", "víctima", "fiscal", "fiscalía",
            "condena", "prisión", "penal", "denuncia", "captura",
            "extorsión", "secuestro", "narcotráfico", "lavado",
            "tentativa", "cómplice", "autor", "sentencia",
        },
        "Laboral": {
            "trabajador", "empleador", "contrato", "salario", "despido",
            "prestaciones", "liquidación", "pensión", "cesantías",
            "vacaciones", "laboral", "trabajo", "jornada", "sindicato",
            "huelga", "patrono", "indemnización", "preaviso",
            "seguridad", "riesgos", "incapacidad", "afiliación",
        },
        "Familia": {
            "divorcio", "custodia", "alimentos", "matrimonio", "adopción",
            "patria", "potestad", "menor", "niño", "adolescente",
            "violencia", "intrafamiliar", "familia", "cónyuge", "hijo",
            "hija", "separación", "unión", "marital", "tutela",
            "curador", "parentesco", "filiación", "conciliación",
        },
    }

    def __init__(
        self,
        umbral_confianza: float = 0.6,
        trazabilidad: Optional[ServicioTrazabilidad] = None,
        xai: Optional[ServicioXAI] = None,
    ):
        """Carga el pipeline de zero-shot classification.

        Inicializa el pipeline de transformers con el modelo mDeBERTa-v3
        para clasificación zero-shot. Si el modelo no está disponible
        localmente, se intentará descargar automáticamente.

        Args:
            umbral_confianza: Umbral mínimo de certeza para considerar
                una predicción como confiable. Rango [0.0, 1.0].
                Por defecto 0.6.
            trazabilidad: Instancia opcional de ServicioTrazabilidad para
                registrar cada clasificación realizada.
            xai: Instancia opcional de ServicioXAI para formatear
                explicaciones y etiquetas de supervisión.
        """
        from transformers import pipeline as hf_pipeline

        self.umbral_confianza = umbral_confianza
        self.trazabilidad = trazabilidad
        self.xai = xai if xai is not None else ServicioXAI()

        # Cargar pipeline de zero-shot classification con mDeBERTa-v3
        try:
            self.pipeline = hf_pipeline(
                "zero-shot-classification",
                model=self.MODELO,
            )
            logger.info("Modelo %s cargado correctamente.", self.MODELO)
        except Exception as e:
            logger.error(
                "Error al cargar el modelo %s: %s. "
                "Verifique su conexión a internet o que el modelo "
                "esté disponible localmente.",
                self.MODELO,
                e,
            )
            raise

    def clasificar(self, texto: str) -> ResultadoClasificacion:
        """Clasifica un texto en una categoría jurídica.

        Ejecuta clasificación zero-shot sobre el texto de entrada,
        asignando exactamente una de las cuatro categorías jurídicas.
        Incluye key tokens para explicabilidad (XAI), nivel de certeza,
        estado de supervisión humana y registro de trazabilidad.

        Si el texto excede los 512 tokens (aproximación por palabras),
        se trunca automáticamente y se registra una advertencia.

        Args:
            texto: Texto legal a clasificar.

        Returns:
            ResultadoClasificacion con categoría, nivel_certeza,
            certeza_porcentaje, key_tokens, requiere_revision,
            estado_supervision y aviso_ia.
        """
        # Truncar texto si excede el límite del modelo
        texto_procesado = self._truncar_texto(texto)

        # Ejecutar clasificación zero-shot
        resultado = self.pipeline(
            texto_procesado,
            candidate_labels=self.CATEGORIAS,
        )

        # Extraer categoría ganadora y su score
        categoria = resultado["labels"][0]
        nivel_certeza = resultado["scores"][0]

        # Extraer key tokens para explicabilidad (XAI)
        key_tokens = self._extraer_key_tokens(texto_procesado, categoria)

        # Formatear explicación XAI
        explicacion = self.xai.formatear_explicacion(
            key_tokens=key_tokens,
            nivel_certeza=nivel_certeza,
            umbral_confianza=self.umbral_confianza,
        )

        # Generar etiquetas de supervisión humana
        supervision = self.xai.etiquetar_supervision(
            nivel_certeza=nivel_certeza,
            umbral_confianza=self.umbral_confianza,
        )

        # Construir resultado de clasificación
        resultado_clasificacion = ResultadoClasificacion(
            texto_entrada=texto,
            categoria=categoria,
            nivel_certeza=nivel_certeza,
            certeza_porcentaje=explicacion["certeza_porcentaje"],
            key_tokens=explicacion["key_tokens"],
            requiere_revision=explicacion["requiere_revision"],
            estado_supervision=supervision["estado"],
            aviso_ia=supervision["aviso"],
        )

        # Registrar en trazabilidad
        if self.trazabilidad is not None:
            self.trazabilidad.registrar_operacion(
                modulo="clasificador",
                entrada=texto,
                resultado={
                    "categoria": categoria,
                    "nivel_certeza": nivel_certeza,
                    "key_tokens": key_tokens,
                },
                version_modelo=self.MODELO,
                metadata={
                    "umbral_confianza": self.umbral_confianza,
                    "requiere_revision": explicacion["requiere_revision"],
                },
            )

        return resultado_clasificacion

    def clasificar_lote(
        self, textos: list[str]
    ) -> list[ResultadoClasificacion]:
        """Clasifica un lote de textos.

        Itera sobre cada texto del lote y aplica la clasificación
        individual. Cada texto se clasifica de forma independiente.

        Args:
            textos: Lista de textos legales a clasificar.

        Returns:
            Lista de ResultadoClasificacion, uno por cada texto de entrada.
        """
        return [self.clasificar(texto) for texto in textos]

    def _extraer_key_tokens(
        self, texto: str, categoria: str
    ) -> list[str]:
        """Extrae tokens relevantes para la decisión del modelo (XAI).

        Utiliza una heurística basada en diccionarios de palabras clave
        por categoría jurídica para identificar los tokens del texto que
        más probablemente influyeron en la clasificación. Los tokens se
        ordenan por relevancia: primero los que coinciden con el
        diccionario de la categoría predicha, luego los tokens más
        largos (que suelen ser más informativos en textos legales).

        Args:
            texto: Texto clasificado.
            categoria: Categoría jurídica asignada por el modelo.

        Returns:
            Lista de tokens relevantes ordenados por relevancia
            descendente. Máximo 10 tokens.
        """
        # Normalizar texto a minúsculas y extraer palabras
        palabras = re.findall(r"[a-záéíóúñü]+", texto.lower())

        # Eliminar duplicados preservando orden de aparición
        palabras_unicas: list[str] = []
        vistas: set[str] = set()
        for palabra in palabras:
            if palabra not in vistas and len(palabra) > 2:
                palabras_unicas.append(palabra)
                vistas.add(palabra)

        # Obtener keywords de la categoría predicha
        keywords_categoria = self._KEYWORDS_POR_CATEGORIA.get(
            categoria, set()
        )

        # Separar en: coincidencias con diccionario vs. otros tokens
        coincidencias = [
            p for p in palabras_unicas if p in keywords_categoria
        ]
        otros = [
            p for p in palabras_unicas if p not in keywords_categoria
        ]

        # Ordenar "otros" por longitud descendente (tokens más largos
        # suelen ser más informativos en textos legales)
        otros.sort(key=len, reverse=True)

        # Combinar: primero coincidencias, luego otros tokens largos
        key_tokens = coincidencias + otros

        # Limitar a máximo 10 tokens
        return key_tokens[:10]

    def _truncar_texto(self, texto: str) -> str:
        """Trunca el texto a ~512 tokens si excede el límite del modelo.

        Utiliza una aproximación basada en palabras (split por espacios)
        para estimar el número de tokens. Si el texto excede el límite,
        se trunca y se registra una advertencia en el log.

        Args:
            texto: Texto de entrada.

        Returns:
            Texto truncado si excedía el límite, o el texto original
            si estaba dentro del límite.
        """
        palabras = texto.split()
        if len(palabras) > self._MAX_TOKENS:
            logger.warning(
                "Texto truncado de %d a %d tokens (palabras).",
                len(palabras),
                self._MAX_TOKENS,
            )
            return " ".join(palabras[: self._MAX_TOKENS])
        return texto
