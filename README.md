# JustIA MVP — Consultorio Jurídico Virtual con IA Responsable

**Autor:** Sergio Fabian Hernandez Vivas  
**Institución:** Corporación Universitaria de Asturias  
**Versión:** 0.1.0

---

## Descripción

JustIA es un Consultorio Jurídico Virtual que utiliza Inteligencia Artificial responsable para fortalecer el acceso a la justicia de poblaciones vulnerables en Colombia: zonas rurales, víctimas del conflicto armado, comunidades indígenas y migrantes.

El MVP integra cinco actividades funcionales ejecutadas de forma secuencial, todas operando bajo principios de **transparencia (XAI)**, **trazabilidad**, **supervisión humana obligatoria** y **privacidad por diseño**, restringidas exclusivamente al marco jurídico colombiano.

## Módulos del Sistema

| Módulo | Descripción | Modelo / Técnica |
|--------|-------------|------------------|
| **Preprocesador** | Normaliza textos legales (minúsculas, limpieza, lematización) | spaCy `es_core_news_sm` |
| **Clasificador** | Clasifica textos en Civil, Penal, Laboral o Familia | mDeBERTa-v3 zero-shot |
| **NER** | Extrae normas legales, jurisdicciones, tipos de violencia y personas | spaCy + EntityRuler |
| **QA** | Responde preguntas sobre normativa colombiana | BERT Spanish-SQuAD |
| **Clustering** | Agrupa casos por similitud temática | TF-IDF + K-Means |

Servicios transversales:

- **ServicioXAI** — Explicabilidad con key tokens, niveles de certeza y supervisión humana obligatoria.
- **ServicioTrazabilidad** — Registro centralizado de operaciones con anonimización de datos personales.

## Estructura del Proyecto

```
proyecto-just-ia/
├── src/
│   └── justia/
│       ├── __init__.py
│       ├── justia_mvp.py        # Punto de entrada principal
│       ├── modelos.py            # Dataclasses del sistema
│       ├── trazabilidad.py       # Servicio de trazabilidad y anonimización
│       ├── xai.py                # Servicio de explicabilidad (XAI)
│       ├── preprocessor.py       # Preprocesamiento de textos legales
│       ├── clasificador.py       # Clasificación temática zero-shot
│       ├── ner.py                # Reconocimiento de entidades nombradas
│       ├── qa.py                 # Preguntas y respuestas con BERT
│       ├── clustering.py         # Agrupamiento temático con K-Means
│       └── data/
│           ├── corpus_legal.json         # Corpus legal sintético (52 registros)
│           └── base_conocimiento.json    # Base normativa colombiana (15 fragmentos)
├── tests/
│   ├── conftest.py               # Fixtures y generadores Hypothesis
│   ├── test_xai.py               # Tests del servicio XAI
│   ├── test_preprocesador.py     # Tests del preprocesador
│   └── test_integracion.py       # Tests de integración entre módulos
├── .gitignore
├── pyproject.toml
├── README.md
└── LICENSE
```

## Requisitos Previos

- Python 3.10 o superior
- pip (gestor de paquetes de Python)

## Instalación

1. Clonar el repositorio:

```bash
git clone https://github.com/tu-usuario/proyecto-just-ia.git
cd proyecto-just-ia
```

2. Instalar el paquete en modo desarrollo (este paso es obligatorio para que el comando `justia` funcione):

```bash
pip install -e ".[dev]"
```

3. Descargar el modelo de spaCy para español:

```bash
python -m spacy download es_core_news_sm
```

## Ejecución

Una vez instalado, ejecutar el MVP completo:

```bash
justia
```

O directamente con Python:

```bash
python -m justia.justia_mvp
```

El sistema ejecutará secuencialmente las cinco actividades:

1. **Preprocesamiento** del corpus legal colombiano
2. **Clasificación temática** de textos de muestra
3. **Extracción de entidades nombradas** (NER)
4. **Respuesta a preguntas legales** frecuentes (QA)
5. **Agrupamiento temático** de casos (Clustering)

Cada actividad genera evidencia de funcionamiento en consola y los registros de trazabilidad se exportan a `registros_trazabilidad.json`.

## Tests

Ejecutar la suite completa de tests (71 tests):

```bash
pytest
```

Con salida detallada:

```bash
pytest -v
```

## Principios de IA Responsable

El sistema opera bajo los siguientes principios:

- **Transparencia (XAI):** Cada predicción incluye los key tokens que influyeron en la decisión del modelo y el nivel de certeza asociado.
- **Supervisión humana obligatoria:** Toda salida se etiqueta como "Pendiente de Validación Humana". Predicciones con baja confianza se escalan a "Prioridad Alta para Revisión".
- **Trazabilidad:** Cada operación genera un registro con timestamp ISO 8601, módulo, versión del modelo, entrada anonimizada y resultado.
- **Privacidad por diseño:** Los datos se procesan exclusivamente en memoria local. Los registros de trazabilidad anonimizan nombres, cédulas, teléfonos y direcciones.
- **Restricción jurisdiccional:** El sistema opera exclusivamente con normativa del marco jurídico colombiano.
- **Sin reconocimiento facial:** El sistema no implementa ni permite funcionalidades de reconocimiento facial.

## Tecnologías Utilizadas

| Tecnología | Uso |
|------------|-----|
| [Transformers](https://huggingface.co/docs/transformers) | Clasificación zero-shot y QA |
| [PyTorch](https://pytorch.org/) | Inferencia con tensores |
| [spaCy](https://spacy.io/) | NER y lematización |
| [scikit-learn](https://scikit-learn.org/) | TF-IDF y K-Means |
| [Hypothesis](https://hypothesis.readthedocs.io/) | Tests basados en propiedades |
| [pytest](https://docs.pytest.org/) | Framework de testing |

## Modelos de IA

| Modelo | Módulo | Fuente |
|--------|--------|--------|
| `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` | Clasificador | [HuggingFace](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli) |
| `mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es` | QA | [HuggingFace](https://huggingface.co/mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es) |
| `es_core_news_sm` | Preprocesador, NER | [spaCy](https://spacy.io/models/es) |

## Autor

**Sergio Fabian Hernandez Vivas**  
Corporación Universitaria de Asturias  
Proyecto de IA Responsable para el Acceso a la Justicia

## Licencia

Este proyecto está licenciado bajo la [Licencia MIT](LICENSE).
