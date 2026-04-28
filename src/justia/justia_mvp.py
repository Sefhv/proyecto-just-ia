"""
Punto de entrada principal del sistema JustIA MVP — Consultorio Jurídico Virtual.

Este módulo orquesta la ejecución secuencial de las cuatro actividades
principales del MVP:

1. **Preprocesamiento** del corpus legal colombiano (normalización y lematización).
2. **Clasificación temática** de textos legales en categorías jurídicas
   (Civil, Penal, Laboral, Familia) mediante mDeBERTa-v3 zero-shot.
3. **Extracción de entidades nombradas** (NER) con spaCy + EntityRuler
   para identificar normas legales, jurisdicciones, tipos de violencia y personas.
4. **Respuesta a preguntas legales** frecuentes (QA) con BERT Spanish-SQuAD,
   restringido a la base de conocimiento de normativa colombiana.
5. **Agrupamiento temático** de casos (Clustering) con TF-IDF + K-Means.

Cada actividad genera evidencia de funcionamiento en consola y registros
de trazabilidad JSON para auditoría. Todos los resultados incluyen
etiquetas de supervisión humana obligatoria y explicabilidad (XAI).

El sistema opera exclusivamente en modo local: los modelos se ejecutan
en memoria sin transmitir datos a servicios externos, cumpliendo con
los principios de privacidad por diseño.

Valida: Requisitos 11.1, 11.2, 11.3, 11.4, 7.3, 9.4
"""

import json
import os
import sys

from justia.trazabilidad import ServicioTrazabilidad
from justia.xai import ServicioXAI


def verificar_dependencias() -> dict:
    """Verifica la disponibilidad de las dependencias requeridas por el sistema.

    Comprueba que las bibliotecas principales (transformers, torch, spacy,
    scikit-learn) estén instaladas y que el modelo spaCy ``es_core_news_sm``
    esté disponible. Informa al usuario sobre las dependencias faltantes
    y sugiere los comandos de instalación correspondientes.

    Returns:
        Diccionario con el estado de cada dependencia, donde las claves
        son los nombres de las dependencias y los valores son ``True``
        si están disponibles o ``False`` si faltan::

            {
                "transformers": True,
                "torch": True,
                "spacy": True,
                "sklearn": True,
                "es_core_news_sm": True,
            }
    """
    dependencias = {
        "transformers": False,
        "torch": False,
        "spacy": False,
        "sklearn": False,
        "es_core_news_sm": False,
    }

    # --- Verificar transformers ---
    try:
        import transformers  # noqa: F401

        dependencias["transformers"] = True
    except ImportError:
        pass

    # --- Verificar torch (PyTorch) ---
    try:
        import torch  # noqa: F401

        dependencias["torch"] = True
    except ImportError:
        pass

    # --- Verificar spacy ---
    try:
        import spacy  # noqa: F401

        dependencias["spacy"] = True
    except ImportError:
        pass

    # --- Verificar scikit-learn ---
    try:
        import sklearn  # noqa: F401

        dependencias["sklearn"] = True
    except ImportError:
        pass

    # --- Verificar modelo spaCy es_core_news_sm ---
    if dependencias["spacy"]:
        try:
            import spacy

            spacy.load("es_core_news_sm")
            dependencias["es_core_news_sm"] = True
        except OSError:
            pass
    # Si spacy no está instalado, el modelo tampoco puede estar disponible

    # --- Informar al usuario sobre dependencias faltantes ---
    faltantes = [dep for dep, disponible in dependencias.items() if not disponible]

    if faltantes:
        print("\n[ADVERTENCIA] Dependencias faltantes detectadas:")
        for dep in faltantes:
            print(f"   [X] {dep}")

        # Separar paquetes pip del modelo spaCy para sugerir comandos claros
        paquetes_pip = []
        necesita_modelo_spacy = False

        for dep in faltantes:
            if dep == "es_core_news_sm":
                necesita_modelo_spacy = True
            elif dep == "sklearn":
                paquetes_pip.append("scikit-learn")
            else:
                paquetes_pip.append(dep)

        print("\n[PAQUETES] Comandos de instalación sugeridos:")
        if paquetes_pip:
            print(f"   pip install {' '.join(paquetes_pip)}")
        if necesita_modelo_spacy:
            print("   python -m spacy download es_core_news_sm")
    else:
        print("\n[OK] Todas las dependencias están disponibles.")

    return dependencias


def ejecutar_actividad(
    nombre: str, funcion: callable, trazabilidad: ServicioTrazabilidad
):
    """Ejecuta una actividad con manejo de errores robusto.

    Envuelve la ejecución de cada actividad del MVP en un bloque
    try/except para que fallas individuales no detengan la ejecución
    de las demás actividades. Registra el resultado (exitoso o fallido)
    en el servicio de trazabilidad.

    Args:
        nombre: Nombre descriptivo de la actividad (e.g. "Clasificación Temática").
        funcion: Callable sin argumentos que ejecuta la actividad y retorna
            su resultado.
        trazabilidad: Instancia de ServicioTrazabilidad para registrar
            el estado de la actividad.

    Returns:
        El resultado de la función si la ejecución fue exitosa, o ``None``
        si ocurrió un error.
    """
    try:
        print(f"\n{'='*60}")
        print(f"Ejecutando: {nombre}")
        print(f"{'='*60}")
        resultado = funcion()
        trazabilidad.registrar_operacion(
            modulo=nombre,
            entrada="actividad_completa",
            resultado={"estado": "exitoso"},
            version_modelo="N/A",
        )
        return resultado
    except ValueError as e:
        print(f"[ADVERTENCIA] Error de validación en {nombre}: {e}")
        trazabilidad.registrar_operacion(
            modulo=nombre,
            entrada="actividad_fallida",
            resultado={"estado": "error_validacion", "detalle": str(e)},
            version_modelo="N/A",
        )
    except Exception as e:
        print(f"[ERROR] Error inesperado en {nombre}: {e}")
        trazabilidad.registrar_operacion(
            modulo=nombre,
            entrada="actividad_fallida",
            resultado={"estado": "error_inesperado", "detalle": str(e)},
            version_modelo="N/A",
        )
    return None


def main():
    """Punto de entrada principal del MVP JustIA.

    Ejecuta las cinco actividades de forma secuencial:
    1. Preprocesamiento del corpus legal
    2. Clasificación temática de textos
    3. Extracción de entidades nombradas
    4. Respuesta a preguntas legales frecuentes
    5. Agrupamiento temático de casos

    Cada actividad genera evidencia de funcionamiento en consola
    y registros de trazabilidad. Fallas individuales en una actividad
    no detienen la ejecución de las demás (manejo robusto de errores).

    Valida: Requisitos 11.1, 11.2, 11.4, 7.3, 9.4
    """

    # ================================================================
    # Banner de bienvenida
    # ================================================================
    print("=" * 60)
    print("  JustIA MVP — Consultorio Jurídico Virtual")
    print("  Corporación Universitaria de Asturias")
    print("  IA Responsable para el Acceso a la Justicia")
    print("=" * 60)
    print()

    # ================================================================
    # 1. Verificación de dependencias
    # Antes de ejecutar cualquier módulo, se comprueba que todas las
    # bibliotecas requeridas estén instaladas y disponibles.
    # ================================================================
    print("[INFO] Verificando dependencias del sistema...")
    dependencias_estado = verificar_dependencias()

    # Si alguna dependencia crítica falta, informar y salir
    faltantes = [dep for dep, ok in dependencias_estado.items() if not ok]
    if faltantes:
        print(
            "\n[ERROR] No se puede continuar sin las dependencias requeridas."
        )
        print("   Instale las dependencias faltantes y vuelva a ejecutar.")
        sys.exit(1)

    # ================================================================
    # 2. Inicialización de servicios transversales
    # ServicioTrazabilidad: registro centralizado de operaciones para
    #   auditoría, con anonimización de datos personales.
    # ServicioXAI: explicabilidad y supervisión humana obligatoria.
    # ================================================================
    print("\n[CONFIG] Inicializando servicios transversales...")
    trazabilidad = ServicioTrazabilidad()
    xai = ServicioXAI()

    # Registrar inicio de sesión con versiones de modelos y dependencias
    # del entorno para garantizar reproducibilidad (Req 7.3).
    import platform

    import sklearn
    import spacy
    import torch
    import transformers

    versiones_modelos = {
        "clasificador": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        "ner": "es_core_news_sm + EntityRuler v1.0",
        "qa": "mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es",
        "clustering": "sklearn TF-IDF + KMeans",
    }

    dependencias_versiones = {
        "python": platform.python_version(),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "spacy": spacy.__version__,
        "scikit-learn": sklearn.__version__,
    }

    trazabilidad.registrar_inicio_sesion(
        versiones_modelos=versiones_modelos,
        dependencias=dependencias_versiones,
    )

    print("   [OK] ServicioTrazabilidad inicializado")
    print("   [OK] ServicioXAI inicializado")
    print(f"   [*] Python {dependencias_versiones['python']}")
    print(f"   [*] Transformers {dependencias_versiones['transformers']}")
    print(f"   [*] PyTorch {dependencias_versiones['torch']}")
    print(f"   [*] spaCy {dependencias_versiones['spacy']}")
    print(f"   [*] scikit-learn {dependencias_versiones['scikit-learn']}")

    # ================================================================
    # 3. Carga de datos
    # Los archivos JSON se ubican en el mismo directorio que este script.
    # Se usa os.path.dirname(os.path.abspath(__file__)) para resolución
    # robusta de rutas independiente del directorio de trabajo actual.
    # ================================================================
    print("\n[DATOS] Cargando datos...")
    directorio_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    # Cargar corpus legal sintético (mínimo 50 registros)
    ruta_corpus = os.path.join(directorio_base, "corpus_legal.json")
    if not os.path.exists(ruta_corpus):
        print(f"[ERROR] No se encontró el archivo: {ruta_corpus}")
        print("   Asegúrese de que corpus_legal.json esté en el directorio del proyecto.")
        sys.exit(1)

    with open(ruta_corpus, "r", encoding="utf-8") as f:
        corpus_crudo = json.load(f)
    print(f"   [OK] Corpus legal cargado: {len(corpus_crudo)} registros")

    # Cargar base de conocimiento normativa colombiana
    ruta_base = os.path.join(directorio_base, "base_conocimiento.json")
    if not os.path.exists(ruta_base):
        print(f"[ERROR] No se encontró el archivo: {ruta_base}")
        print("   Asegúrese de que base_conocimiento.json esté en el directorio del proyecto.")
        sys.exit(1)

    with open(ruta_base, "r", encoding="utf-8") as f:
        base_conocimiento = json.load(f)
    print(f"   [OK] Base de conocimiento cargada: {len(base_conocimiento)} fragmentos normativos")

    # ================================================================
    # Variable compartida para almacenar el corpus procesado entre
    # actividades. Se usa una lista mutable para permitir asignación
    # desde las funciones lambda de cada actividad.
    # ================================================================
    corpus_procesado = []

    # ================================================================
    # ACTIVIDAD 1: Preprocesamiento del corpus legal
    # Normaliza textos legales crudos (minúsculas, eliminación de
    # caracteres especiales, lematización con spaCy) para que los
    # módulos de IA reciban datos limpios y consistentes.
    # ================================================================
    def actividad_preprocesamiento():
        from justia.preprocessor import Preprocesador

        print("  Inicializando Preprocesador con spaCy es_core_news_sm...")
        preprocesador = Preprocesador(trazabilidad=trazabilidad)

        print(f"  Procesando {len(corpus_crudo)} registros del corpus legal...")
        resultado = preprocesador.procesar_corpus(corpus_crudo)

        # Almacenar resultado para uso en actividades posteriores
        corpus_procesado.clear()
        corpus_procesado.extend(resultado)

        print(f"\n  [RESULTADO] Resultados del preprocesamiento:")
        print(f"     Registros procesados: {len(resultado)} de {len(corpus_crudo)}")

        # Mostrar muestra de los primeros 3 registros procesados
        if resultado:
            print(f"\n  [MUESTRA] Muestra de registros procesados (primeros 3):")
            for i, reg in enumerate(resultado[:3]):
                print(f"\n     --- Registro {i + 1} ---")
                print(f"     Original: {reg['original'][:100]}...")
                print(f"     Limpio:   {reg['limpio'][:100]}...")

        return resultado

    ejecutar_actividad(
        "Actividad 1: Preprocesamiento del Corpus Legal",
        actividad_preprocesamiento,
        trazabilidad,
    )

    # ================================================================
    # ACTIVIDAD 2: Clasificación temática de textos legales
    # Clasifica textos en una de cuatro categorías jurídicas
    # (Civil, Penal, Laboral, Familia) usando mDeBERTa-v3 zero-shot.
    # Incluye key tokens (XAI) y supervisión humana obligatoria.
    # ================================================================
    def actividad_clasificacion():
        from justia.clasificador import ModuloClasificador

        print("  Inicializando ModuloClasificador con mDeBERTa-v3...")
        clasificador = ModuloClasificador(
            trazabilidad=trazabilidad, xai=xai
        )

        # Seleccionar 3-5 textos de muestra del corpus procesado
        # Se usan los textos originales para clasificación más precisa
        textos_muestra = corpus_procesado[:5] if corpus_procesado else []
        if not textos_muestra:
            print("  [ADVERTENCIA] No hay textos procesados disponibles para clasificar.")
            return []

        print(f"  Clasificando {len(textos_muestra)} textos de muestra...\n")
        resultados = []
        for i, registro in enumerate(textos_muestra):
            texto = registro["original"]
            resultado = clasificador.clasificar(texto)
            resultados.append(resultado)

            print(f"  [*] Texto {i + 1}:")
            print(f"     Entrada:     {texto[:80]}...")
            print(f"     Categoría:   {resultado.categoria}")
            print(f"     Certeza:     {resultado.certeza_porcentaje}")
            print(f"     Key Tokens:  {', '.join(resultado.key_tokens[:5])}")
            print(f"     Supervisión: {resultado.estado_supervision}")
            print(f"     [IA] {resultado.aviso_ia}")
            print()

        return resultados

    ejecutar_actividad(
        "Actividad 2: Clasificación Temática de Textos",
        actividad_clasificacion,
        trazabilidad,
    )

    # ================================================================
    # ACTIVIDAD 3: Extracción de entidades nombradas (NER)
    # Identifica normas legales, jurisdicciones, tipos de violencia
    # y personas en textos jurídicos usando spaCy + EntityRuler.
    # ================================================================
    def actividad_ner():
        from justia.ner import ModuloNER

        print("  Inicializando ModuloNER con spaCy + EntityRuler...")
        modulo_ner = ModuloNER(trazabilidad=trazabilidad, xai=xai)

        # Seleccionar 3-5 textos de muestra del corpus procesado
        # Se usan los textos originales para preservar entidades
        textos_muestra = corpus_procesado[:5] if corpus_procesado else []
        if not textos_muestra:
            print("  [ADVERTENCIA] No hay textos procesados disponibles para NER.")
            return []

        print(f"  Extrayendo entidades de {len(textos_muestra)} textos de muestra...\n")
        resultados = []
        for i, registro in enumerate(textos_muestra):
            texto = registro["original"]
            resultado = modulo_ner.extraer_entidades(texto)
            resultados.append(resultado)

            print(f"  [*] Texto {i + 1}: {texto[:80]}...")
            print(f"     Total entidades: {resultado.total_entidades}")

            if resultado.entidades:
                for ent in resultado.entidades:
                    print(
                        f"     -> [{ent.etiqueta}] \"{ent.texto}\" "
                        f"(pos: {ent.inicio}-{ent.fin})"
                    )
            elif resultado.mensaje:
                print(f"     [INFO] {resultado.mensaje}")

            print(f"     Supervisión: {resultado.estado_supervision}")
            print(f"     [IA] {resultado.aviso_ia}")
            print()

        return resultados

    ejecutar_actividad(
        "Actividad 3: Extracción de Entidades Nombradas (NER)",
        actividad_ner,
        trazabilidad,
    )

    # ================================================================
    # ACTIVIDAD 4: Respuesta a preguntas legales frecuentes (QA)
    # Responde preguntas sobre normativa colombiana usando BERT
    # Spanish-SQuAD, restringido a la base de conocimiento local.
    # ================================================================
    def actividad_qa():
        from justia.qa import ModuloQA

        print("  Inicializando ModuloQA con BERT Spanish-SQuAD...")
        modulo_qa = ModuloQA(
            base_conocimiento=base_conocimiento,
            trazabilidad=trazabilidad,
            xai=xai,
        )

        # Preguntas de muestra sobre derecho colombiano
        preguntas = [
            "¿Qué protección tienen los menores de edad?",
            "¿Cuáles son las causales de terminación del contrato de trabajo?",
            "¿Qué es la violencia contra la mujer?",
        ]

        print(f"  Respondiendo {len(preguntas)} preguntas de muestra...\n")
        resultados = []
        for i, pregunta in enumerate(preguntas):
            resultado = modulo_qa.responder(pregunta)
            resultados.append(resultado)

            print(f"  [*] Pregunta {i + 1}: {pregunta}")
            print(f"     Respuesta:   {resultado.respuesta[:120]}...")
            print(f"     Fuente:      {resultado.fuente_normativa}")
            print(f"     Certeza:     {resultado.certeza_porcentaje}")
            print(f"     Key Tokens:  {', '.join(resultado.key_tokens[:5])}")
            print(f"     Supervisión: {resultado.estado_supervision}")
            print(f"     [IA] {resultado.aviso_ia}")
            print()

        return resultados

    ejecutar_actividad(
        "Actividad 4: Respuesta a Preguntas Legales (QA)",
        actividad_qa,
        trazabilidad,
    )

    # ================================================================
    # ACTIVIDAD 5: Agrupamiento temático de casos (Clustering)
    # Agrupa casos por similitud temática usando TF-IDF + K-Means
    # para identificar tendencias y patrones de vulnerabilidad.
    # ================================================================
    def actividad_clustering():
        from justia.clustering import ModuloClustering

        print("  Inicializando ModuloClustering con TF-IDF + K-Means...")
        modulo_clustering = ModuloClustering(
            trazabilidad=trazabilidad, xai=xai
        )

        # Extraer textos limpios del corpus procesado para clustering
        textos_limpios = [
            reg["limpio"] for reg in corpus_procesado if reg.get("limpio")
        ]

        if len(textos_limpios) < 50:
            print(
                f"  [ADVERTENCIA] Se requieren al menos 50 textos para clustering. "
                f"Disponibles: {len(textos_limpios)}"
            )
            raise ValueError(
                f"Insuficientes textos para clustering: {len(textos_limpios)} < 50"
            )

        print(f"  Agrupando {len(textos_limpios)} textos en clusters temáticos...\n")
        resultado = modulo_clustering.agrupar(textos_limpios)

        print(f"  [RESULTADO] Resultados del agrupamiento:")
        print(f"     Clusters generados: {resultado.num_clusters}")
        print(f"     Total de casos:     {resultado.num_casos_total}")
        print(f"     Inercia:            {resultado.inercia:.2f}")
        print()

        for cluster in resultado.clusters:
            print(f"  [*] Cluster {cluster.cluster_id}:")
            print(f"     Casos asignados:  {cluster.num_casos}")
            print(f"     Palabras clave:   {', '.join(cluster.palabras_clave)}")
            print(f"     Resumen:          {cluster.resumen}")
            print()

        print(f"     Supervisión: {resultado.estado_supervision}")
        print(f"     [IA] {resultado.aviso_ia}")

        return resultado

    ejecutar_actividad(
        "Actividad 5: Agrupamiento Temático de Casos (Clustering)",
        actividad_clustering,
        trazabilidad,
    )

    # ================================================================
    # 4. Exportación de registros de trazabilidad
    # Se exportan todos los registros acumulados durante la sesión
    # a un archivo JSON estructurado para auditoría (Req 7.2, 7.4).
    # ================================================================
    print(f"\n{'='*60}")
    print("[EXPORTAR] Exportando registros de trazabilidad...")
    print(f"{'='*60}")

    ruta_registros = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "registros_trazabilidad.json"
    )
    try:
        trazabilidad.exportar_registros(ruta=ruta_registros)
        print(f"   [OK] Registros exportados a: {ruta_registros}")
        print(f"   [RESULTADO] Total de operaciones registradas: {len(trazabilidad.sesion.registros)}")
    except Exception as e:
        print(f"   [ERROR] Error al exportar registros: {e}")

    # ================================================================
    # 5. Resumen final de la ejecución
    # ================================================================
    print(f"\n{'='*60}")
    print("  [OK] Ejecución del MVP JustIA completada")
    print(f"{'='*60}")
    print(f"  Operaciones registradas: {len(trazabilidad.sesion.registros)}")
    print(f"  Archivo de trazabilidad: registros_trazabilidad.json")
    print()
    print(f"  [IA] {xai.AVISO_IA}")
    print()
    print("  Nota: Este sistema NO utiliza reconocimiento facial (Req 9.4).")
    print("  Todos los datos se procesan localmente en memoria.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
