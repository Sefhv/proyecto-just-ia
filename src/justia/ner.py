"""
Módulo NER (Reconocimiento de Entidades Nombradas) del sistema JustIA MVP.

Extrae entidades legales colombianas de textos jurídicos usando spaCy con
EntityRuler personalizado. Identifica normas legales, jurisdicciones,
tipos de violencia y personas, integrando explicabilidad (XAI),
supervisión humana obligatoria y trazabilidad de cada operación.

Valida: Requisitos 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7
"""

import logging
from typing import Optional

import spacy

from justia.modelos import EntidadLegal, ResultadoNER
from justia.trazabilidad import ServicioTrazabilidad
from justia.xai import ServicioXAI

logger = logging.getLogger(__name__)


class ModuloNER:
    """Reconocimiento de Entidades Nombradas para textos jurídicos colombianos.

    Utiliza spaCy con un EntityRuler personalizado para identificar cuatro
    categorías de entidades legales: normas legales (leyes, decretos,
    artículos, códigos), jurisdicciones (cortes, tribunales, juzgados),
    tipos de violencia y personas. Cada extracción incluye posiciones
    exactas en el texto, estado de supervisión humana y registro de
    trazabilidad.

    Attributes:
        ETIQUETAS: Las cuatro categorías de entidades legales válidas.
        nlp: Modelo spaCy con EntityRuler configurado.
        trazabilidad: Servicio de trazabilidad para registrar operaciones.
        xai: Servicio de explicabilidad para etiquetas de supervisión.
    """

    ETIQUETAS = ["NORMA_LEGAL", "JURISDICCION", "VIOLENCIA_TIPO", "PERSONA"]

    # Versión del módulo NER para trazabilidad
    _VERSION = "es_core_news_sm + EntityRuler v1.0"

    def __init__(
        self,
        nlp_model: str = "es_core_news_sm",
        trazabilidad: Optional[ServicioTrazabilidad] = None,
        xai: Optional[ServicioXAI] = None,
    ):
        """Carga spaCy y configura EntityRuler con patrones legales colombianos.

        Inicializa el modelo spaCy y agrega un EntityRuler con patrones
        personalizados para el contexto jurídico colombiano. El EntityRuler
        se inserta antes del componente NER estándar para que los patrones
        personalizados tengan prioridad sobre el reconocimiento base.

        Args:
            nlp_model: Nombre del modelo spaCy a cargar. Por defecto
                ``es_core_news_sm`` (español).
            trazabilidad: Instancia opcional de ServicioTrazabilidad para
                registrar cada extracción de entidades.
            xai: Instancia opcional de ServicioXAI para generar etiquetas
                de supervisión humana.

        Raises:
            OSError: Si el modelo spaCy no está instalado.
        """
        self.trazabilidad = trazabilidad
        self.xai = xai if xai is not None else ServicioXAI()

        # Cargar modelo spaCy
        self.nlp = spacy.load(nlp_model)
        logger.info("Modelo spaCy '%s' cargado correctamente.", nlp_model)

        # Configurar EntityRuler con patrones legales colombianos
        self._configurar_entity_ruler(self.nlp)

    def _configurar_entity_ruler(self, nlp) -> None:
        """Agrega patrones personalizados al EntityRuler.

        Configura patrones para cuatro categorías de entidades legales
        colombianas:

        - **NORMA_LEGAL**: "Ley [número] de [año]", "Decreto [número]",
          "Artículo [número]", "Código [nombre]".
        - **JURISDICCION**: Corte Constitucional, Corte Suprema de Justicia,
          Tribunal Superior, Juzgado [ordinal] [tipo], Fiscalía General
          de la Nación, Consejo de Estado, etc.
        - **VIOLENCIA_TIPO**: violencia intrafamiliar, patrimonial,
          económica, física, psicológica, sexual.
        - **PERSONA**: Patrones de nombres propios en contexto legal
          (demandante, demandado, señor/señora + nombre).

        El EntityRuler se inserta antes del componente NER estándar
        para que los patrones personalizados tengan prioridad.

        Args:
            nlp: Instancia del modelo spaCy donde agregar el EntityRuler.
        """
        # Crear EntityRuler antes del NER estándar para priorizar patrones
        ruler = nlp.add_pipe("entity_ruler", before="ner")

        patterns = []

        # --- NORMA_LEGAL: Leyes, decretos, artículos y códigos ---

        # "Ley [número] de [año]" — patrón token-based
        patterns.append({
            "label": "NORMA_LEGAL",
            "pattern": [
                {"LOWER": "ley"},
                {"IS_DIGIT": True},
                {"LOWER": "de"},
                {"IS_DIGIT": True},
            ],
        })

        # "Decreto [número]"
        patterns.append({
            "label": "NORMA_LEGAL",
            "pattern": [
                {"LOWER": "decreto"},
                {"IS_DIGIT": True},
            ],
        })

        # "Decreto [número] de [año]"
        patterns.append({
            "label": "NORMA_LEGAL",
            "pattern": [
                {"LOWER": "decreto"},
                {"IS_DIGIT": True},
                {"LOWER": "de"},
                {"IS_DIGIT": True},
            ],
        })

        # "Artículo [número]" (con y sin tilde)
        patterns.append({
            "label": "NORMA_LEGAL",
            "pattern": [
                {"LOWER": "artículo"},
                {"IS_DIGIT": True},
            ],
        })
        patterns.append({
            "label": "NORMA_LEGAL",
            "pattern": [
                {"LOWER": "articulo"},
                {"IS_DIGIT": True},
            ],
        })

        # "Art. [número]"
        patterns.append({
            "label": "NORMA_LEGAL",
            "pattern": [
                {"LOWER": "art"},
                {"TEXT": "."},
                {"IS_DIGIT": True},
            ],
        })

        # "Código Civil"
        patterns.append({
            "label": "NORMA_LEGAL",
            "pattern": [
                {"LOWER": "código"},
                {"LOWER": "civil"},
            ],
        })

        # "Código Penal"
        patterns.append({
            "label": "NORMA_LEGAL",
            "pattern": [
                {"LOWER": "código"},
                {"LOWER": "penal"},
            ],
        })

        # "Código de la Infancia y la Adolescencia"
        patterns.append({
            "label": "NORMA_LEGAL",
            "pattern": [
                {"LOWER": "código"},
                {"LOWER": "de"},
                {"LOWER": "la"},
                {"LOWER": "infancia"},
                {"LOWER": "y"},
                {"LOWER": "la"},
                {"LOWER": "adolescencia"},
            ],
        })

        # "Código Sustantivo del Trabajo"
        patterns.append({
            "label": "NORMA_LEGAL",
            "pattern": [
                {"LOWER": "código"},
                {"LOWER": "sustantivo"},
                {"LOWER": "del"},
                {"LOWER": "trabajo"},
            ],
        })

        # "Código General del Proceso"
        patterns.append({
            "label": "NORMA_LEGAL",
            "pattern": [
                {"LOWER": "código"},
                {"LOWER": "general"},
                {"LOWER": "del"},
                {"LOWER": "proceso"},
            ],
        })

        # "Código de Procedimiento Penal"
        patterns.append({
            "label": "NORMA_LEGAL",
            "pattern": [
                {"LOWER": "código"},
                {"LOWER": "de"},
                {"LOWER": "procedimiento"},
                {"LOWER": "penal"},
            ],
        })

        # "Constitución Política" (con y sin tildes)
        patterns.append({
            "label": "NORMA_LEGAL",
            "pattern": [
                {"LOWER": "constitución"},
                {"LOWER": "política"},
            ],
        })
        patterns.append({
            "label": "NORMA_LEGAL",
            "pattern": [
                {"LOWER": "constitucion"},
                {"LOWER": "politica"},
            ],
        })

        # --- JURISDICCION: Cortes, tribunales, juzgados ---

        # "Corte Constitucional"
        patterns.append({
            "label": "JURISDICCION",
            "pattern": [
                {"LOWER": "corte"},
                {"LOWER": "constitucional"},
            ],
        })

        # "Corte Suprema de Justicia"
        patterns.append({
            "label": "JURISDICCION",
            "pattern": [
                {"LOWER": "corte"},
                {"LOWER": "suprema"},
                {"LOWER": "de"},
                {"LOWER": "justicia"},
            ],
        })

        # "Tribunal Superior"
        patterns.append({
            "label": "JURISDICCION",
            "pattern": [
                {"LOWER": "tribunal"},
                {"LOWER": "superior"},
            ],
        })

        # "Tribunal Superior de [ciudad]"
        patterns.append({
            "label": "JURISDICCION",
            "pattern": [
                {"LOWER": "tribunal"},
                {"LOWER": "superior"},
                {"LOWER": "de"},
                {"IS_TITLE": True},
            ],
        })

        # "Consejo de Estado"
        patterns.append({
            "label": "JURISDICCION",
            "pattern": [
                {"LOWER": "consejo"},
                {"LOWER": "de"},
                {"LOWER": "estado"},
            ],
        })

        # "Consejo Superior de la Judicatura"
        patterns.append({
            "label": "JURISDICCION",
            "pattern": [
                {"LOWER": "consejo"},
                {"LOWER": "superior"},
                {"LOWER": "de"},
                {"LOWER": "la"},
                {"LOWER": "judicatura"},
            ],
        })

        # "Fiscalía General de la Nación" (con y sin tildes)
        patterns.append({
            "label": "JURISDICCION",
            "pattern": [
                {"LOWER": "fiscalía"},
                {"LOWER": "general"},
                {"LOWER": "de"},
                {"LOWER": "la"},
                {"LOWER": "nación"},
            ],
        })
        patterns.append({
            "label": "JURISDICCION",
            "pattern": [
                {"LOWER": "fiscalia"},
                {"LOWER": "general"},
                {"LOWER": "de"},
                {"LOWER": "la"},
                {"LOWER": "nacion"},
            ],
        })

        # "Procuraduría General de la Nación" (con y sin tildes)
        patterns.append({
            "label": "JURISDICCION",
            "pattern": [
                {"LOWER": "procuraduría"},
                {"LOWER": "general"},
                {"LOWER": "de"},
                {"LOWER": "la"},
                {"LOWER": "nación"},
            ],
        })
        patterns.append({
            "label": "JURISDICCION",
            "pattern": [
                {"LOWER": "procuraduria"},
                {"LOWER": "general"},
                {"LOWER": "de"},
                {"LOWER": "la"},
                {"LOWER": "nacion"},
            ],
        })

        # "Defensoría del Pueblo" (con y sin tilde)
        patterns.append({
            "label": "JURISDICCION",
            "pattern": [
                {"LOWER": "defensoría"},
                {"LOWER": "del"},
                {"LOWER": "pueblo"},
            ],
        })
        patterns.append({
            "label": "JURISDICCION",
            "pattern": [
                {"LOWER": "defensoria"},
                {"LOWER": "del"},
                {"LOWER": "pueblo"},
            ],
        })

        # "Juzgado [ordinal] [tipo]" — e.g. "Juzgado Primero Civil"
        for ordinal in [
            "primero", "segundo", "tercero", "cuarto", "quinto",
            "sexto", "séptimo", "septimo", "octavo", "noveno", "décimo", "decimo",
        ]:
            for tipo in ["civil", "penal", "laboral", "familia", "promiscuo"]:
                patterns.append({
                    "label": "JURISDICCION",
                    "pattern": [
                        {"LOWER": "juzgado"},
                        {"LOWER": ordinal},
                        {"LOWER": tipo},
                    ],
                })

        # "Juzgado [ordinal] [tipo] Municipal"
        for ordinal in [
            "primero", "segundo", "tercero", "cuarto", "quinto",
        ]:
            for tipo in ["civil", "penal", "laboral", "promiscuo"]:
                patterns.append({
                    "label": "JURISDICCION",
                    "pattern": [
                        {"LOWER": "juzgado"},
                        {"LOWER": ordinal},
                        {"LOWER": tipo},
                        {"LOWER": "municipal"},
                    ],
                })

        # "Juzgado de Familia"
        patterns.append({
            "label": "JURISDICCION",
            "pattern": [
                {"LOWER": "juzgado"},
                {"LOWER": "de"},
                {"LOWER": "familia"},
            ],
        })

        # --- VIOLENCIA_TIPO: Tipos de violencia ---

        for tipo_violencia in [
            "intrafamiliar", "patrimonial", "económica", "economica",
            "física", "fisica", "psicológica", "psicologica", "sexual",
        ]:
            patterns.append({
                "label": "VIOLENCIA_TIPO",
                "pattern": [
                    {"LOWER": "violencia"},
                    {"LOWER": tipo_violencia},
                ],
            })

        # --- PERSONA: Patrones de nombres propios en contexto legal ---
        # Se complementa con el mapeo PER -> PERSONA del NER base de spaCy.
        # Aquí se agregan patrones contextuales para roles legales.

        for rol in ["demandante", "demandado", "acusado", "víctima"]:
            # "el/la [rol]" — estos se capturan como PERSONA
            patterns.append({
                "label": "PERSONA",
                "pattern": [
                    {"LOWER": {"IN": ["el", "la"]}},
                    {"LOWER": rol},
                ],
            })

        # "señor/señora [Nombre]"
        patterns.append({
            "label": "PERSONA",
            "pattern": [
                {"LOWER": {"IN": ["señor", "señora"]}},
                {"IS_TITLE": True},
            ],
        })

        # "señor/señora [Nombre] [Apellido]"
        patterns.append({
            "label": "PERSONA",
            "pattern": [
                {"LOWER": {"IN": ["señor", "señora"]}},
                {"IS_TITLE": True},
                {"IS_TITLE": True},
            ],
        })

        # Agregar todos los patrones al ruler
        ruler.add_patterns(patterns)
        logger.info(
            "EntityRuler configurado con %d patrones legales colombianos.",
            len(patterns),
        )

    def extraer_entidades(self, texto: str) -> ResultadoNER:
        """Extrae entidades legales de un texto jurídico.

        Procesa el texto con spaCy y el EntityRuler configurado,
        recopilando entidades que coincidan con las etiquetas legales
        definidas. Las entidades PER del NER base de spaCy se mapean
        a PERSONA para mantener consistencia con las etiquetas del sistema.

        Si no se encuentran entidades, se retorna un mensaje informativo
        sugiriendo reformular la consulta.

        Args:
            texto: Texto jurídico a analizar.

        Returns:
            ResultadoNER con lista de EntidadLegal (texto, etiqueta,
            inicio, fin), total de entidades, estado de supervisión
            humana y aviso de IA.
        """
        # Procesar texto con spaCy (EntityRuler + NER base)
        doc = self.nlp(texto)

        # Recopilar entidades que coincidan con nuestras etiquetas
        entidades = []
        for ent in doc.ents:
            # Mapear PER (spaCy) -> PERSONA (JustIA)
            etiqueta = "PERSONA" if ent.label_ == "PER" else ent.label_

            # Solo incluir entidades con etiquetas válidas del sistema
            if etiqueta in self.ETIQUETAS:
                entidades.append(
                    EntidadLegal(
                        texto=ent.text,
                        etiqueta=etiqueta,
                        inicio=ent.start_char,
                        fin=ent.end_char,
                    )
                )

        # Generar etiquetas de supervisión humana
        supervision = self.xai.etiquetar_supervision(
            nivel_certeza=1.0,  # NER basado en reglas: certeza alta
            umbral_confianza=0.6,
        )

        # Determinar mensaje informativo si no se encontraron entidades
        mensaje = None
        if not entidades:
            mensaje = (
                "No se identificaron entidades legales en el texto proporcionado. "
                "Sugerencia: reformule la consulta incluyendo referencias a leyes, "
                "instituciones judiciales o tipos de violencia específicos."
            )

        # Construir resultado NER
        resultado = ResultadoNER(
            texto_entrada=texto,
            entidades=entidades,
            total_entidades=len(entidades),
            mensaje=mensaje,
            estado_supervision=supervision["estado"],
            aviso_ia=supervision["aviso"],
        )

        # Registrar en trazabilidad
        if self.trazabilidad is not None:
            self.trazabilidad.registrar_operacion(
                modulo="ner",
                entrada=texto,
                resultado={
                    "total_entidades": len(entidades),
                    "entidades": [
                        {
                            "texto": e.texto,
                            "etiqueta": e.etiqueta,
                            "inicio": e.inicio,
                            "fin": e.fin,
                        }
                        for e in entidades
                    ],
                    "mensaje": mensaje,
                },
                version_modelo=self._VERSION,
                metadata={
                    "etiquetas_encontradas": list(
                        {e.etiqueta for e in entidades}
                    ),
                },
            )

        return resultado
