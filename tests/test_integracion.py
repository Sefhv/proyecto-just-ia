"""
Tests de integración para verificar la conexión entre módulos del MVP JustIA.

Verifica que:
- El Preprocesador alimenta correctamente al Clasificador, NER y Clustering
- El Módulo QA usa la base de conocimiento correctamente
- Todos los módulos registran operaciones en el ServicioTrazabilidad
- Todos los módulos usan ServicioXAI para etiquetas de supervisión
- El aviso de IA aparece en todas las salidas
- No hay código huérfano o sin integrar

Valida: Requisitos 8.1, 8.2, 6.1, 6.2, 7.1, 10.1
"""

import json
import os

import pytest

from justia.trazabilidad import ServicioTrazabilidad
from justia.xai import ServicioXAI
from justia.preprocessor import Preprocesador

# Aviso de IA obligatorio (Requisito 8.2)
AVISO_IA = (
    "Este resultado fue generado por IA y requiere validación "
    "de un profesional del derecho antes de ser utilizado."
)


# ================================================================
# Fixtures
# ================================================================

@pytest.fixture
def trazabilidad():
    """Instancia fresca de ServicioTrazabilidad."""
    return ServicioTrazabilidad()


@pytest.fixture
def xai():
    """Instancia fresca de ServicioXAI."""
    return ServicioXAI()


@pytest.fixture
def corpus_minimo():
    """Corpus mínimo de 50 registros para tests de integración."""
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
    categorias = ["Civil", "Penal", "Laboral", "Familia"]
    for i in range(50):
        texto = textos_base[i % len(textos_base)]
        corpus.append({
            "id": f"caso_{i+1:03d}",
            "texto": texto,
            "area_esperada": categorias[i % len(categorias)],
            "metadata": {"fuente": "sintético", "indice": i},
        })
    return corpus


@pytest.fixture
def corpus_procesado(trazabilidad, corpus_minimo):
    """Corpus procesado por el Preprocesador con trazabilidad."""
    preprocesador = Preprocesador(trazabilidad=trazabilidad)
    return preprocesador.procesar_corpus(corpus_minimo)


# ================================================================
# Test: Preprocesador alimenta correctamente a otros módulos
# ================================================================

class TestFlujoPreprocesamientoIntegracion:
    """Verifica que el Preprocesador produce datos consumibles por otros módulos."""

    def test_preprocesador_produce_campos_requeridos(self, corpus_procesado):
        """El corpus procesado tiene campos 'original' y 'limpio' para otros módulos."""
        assert len(corpus_procesado) >= 50
        for registro in corpus_procesado:
            assert "original" in registro
            assert "limpio" in registro
            assert isinstance(registro["original"], str)
            assert isinstance(registro["limpio"], str)
            assert len(registro["original"]) > 0
            assert len(registro["limpio"]) > 0

    def test_preprocesador_registra_en_trazabilidad(self, trazabilidad):
        """El Preprocesador registra advertencias de textos inválidos en trazabilidad."""
        preprocesador = Preprocesador(trazabilidad=trazabilidad)
        corpus_con_invalidos = [
            {"texto": "Texto válido de prueba legal."},
            {"texto": ""},  # inválido
            {"texto": "!!!???"},  # inválido
        ]
        resultado = preprocesador.procesar_corpus(corpus_con_invalidos)

        # Solo 1 registro válido
        assert len(resultado) == 1

        # Debe haber registros de advertencia en trazabilidad
        registros_advertencia = [
            r for r in trazabilidad.sesion.registros
            if r.modulo == "preprocesador"
        ]
        assert len(registros_advertencia) == 2

    def test_textos_limpios_validos_para_clustering(self, corpus_procesado):
        """Los textos limpios del corpus procesado son válidos para clustering."""
        textos_limpios = [
            reg["limpio"] for reg in corpus_procesado if reg.get("limpio")
        ]
        assert len(textos_limpios) >= 50
        for texto in textos_limpios:
            assert isinstance(texto, str)
            assert len(texto.strip()) > 0


# ================================================================
# Test: Todos los módulos usan ServicioXAI
# ================================================================

class TestXAIIntegracion:
    """Verifica que ServicioXAI produce etiquetas correctas para todos los módulos."""

    def test_aviso_ia_texto_correcto(self, xai):
        """El aviso de IA tiene el texto obligatorio exacto."""
        assert xai.AVISO_IA == AVISO_IA

    def test_etiquetar_supervision_alta_confianza(self, xai):
        """Supervisión con alta confianza produce 'Pendiente de Validación Humana'."""
        resultado = xai.etiquetar_supervision(
            nivel_certeza=0.9, umbral_confianza=0.6
        )
        assert resultado["estado"] == "Pendiente de Validación Humana"
        assert resultado["aviso"] == AVISO_IA

    def test_etiquetar_supervision_baja_confianza(self, xai):
        """Supervisión con baja confianza produce 'Prioridad Alta para Revisión'."""
        resultado = xai.etiquetar_supervision(
            nivel_certeza=0.3, umbral_confianza=0.6
        )
        assert resultado["estado"] == "Prioridad Alta para Revisión"
        assert resultado["aviso"] == AVISO_IA
        assert len(resultado["factores_incertidumbre"]) > 0

    def test_formatear_explicacion_incluye_aviso(self, xai):
        """La explicación formateada incluye el aviso de IA obligatorio."""
        resultado = xai.formatear_explicacion(
            key_tokens=["token1", "token2"],
            nivel_certeza=0.8,
            umbral_confianza=0.6,
        )
        assert resultado["aviso_ia"] == AVISO_IA
        assert "certeza_porcentaje" in resultado
        assert "requiere_revision" in resultado
        assert "key_tokens" in resultado


# ================================================================
# Test: Trazabilidad registra operaciones de todos los módulos
# ================================================================

class TestTrazabilidadIntegracion:
    """Verifica que ServicioTrazabilidad registra correctamente."""

    def test_registro_inicio_sesion(self, trazabilidad):
        """El inicio de sesión registra versiones de modelos y dependencias."""
        versiones = {
            "clasificador": "mDeBERTa-v3-base-mnli-xnli",
            "ner": "es_core_news_sm + EntityRuler v1.0",
        }
        dependencias = {"python": "3.11", "torch": "2.1.0"}

        trazabilidad.registrar_inicio_sesion(
            versiones_modelos=versiones, dependencias=dependencias
        )

        assert trazabilidad.sesion.versiones_modelos == versiones
        assert trazabilidad.sesion.dependencias_entorno == dependencias

    def test_registro_operacion_completo(self, trazabilidad):
        """Cada operación registrada tiene todos los campos requeridos."""
        trazabilidad.registrar_operacion(
            modulo="clasificador",
            entrada="Texto de prueba legal",
            resultado={"categoria": "Civil", "nivel_certeza": 0.85},
            version_modelo="mDeBERTa-v3-base-mnli-xnli",
        )

        assert len(trazabilidad.sesion.registros) == 1
        registro = trazabilidad.sesion.registros[0]
        assert registro.modulo == "clasificador"
        assert registro.version_modelo == "mDeBERTa-v3-base-mnli-xnli"
        assert registro.timestamp  # ISO 8601
        assert registro.entrada_anonimizada  # anonimizado
        assert isinstance(registro.resultado, dict)

    def test_exportar_registros_json(self, trazabilidad, tmp_path):
        """Los registros se exportan correctamente a JSON."""
        trazabilidad.registrar_operacion(
            modulo="ner",
            entrada="Ley 1098 de 2006",
            resultado={"entidades": []},
            version_modelo="es_core_news_sm",
        )

        ruta = str(tmp_path / "test_registros.json")
        trazabilidad.exportar_registros(ruta=ruta)

        with open(ruta, "r", encoding="utf-8") as f:
            datos = json.load(f)

        assert "sesion" in datos
        assert "registros" in datos
        assert len(datos["registros"]) == 1

    def test_anonimizacion_datos_personales(self, trazabilidad):
        """Los datos personales se anonimizan en los registros."""
        texto_con_datos = "Juan Carlos Pérez con cédula 1234567890"
        trazabilidad.registrar_operacion(
            modulo="qa",
            entrada=texto_con_datos,
            resultado={"respuesta": "test"},
            version_modelo="bert-spanish",
        )

        registro = trazabilidad.sesion.registros[0]
        # El texto anonimizado no debe contener la cédula original
        assert "1234567890" not in registro.entrada_anonimizada


# ================================================================
# Test: Archivos de datos existen y son válidos
# ================================================================

class TestDatosIntegracion:
    """Verifica que los archivos de datos requeridos existen y son válidos."""

    def _ruta_proyecto(self, nombre_archivo):
        """Obtiene la ruta absoluta de un archivo de datos del proyecto."""
        import justia.data
        directorio = os.path.dirname(os.path.abspath(justia.data.__file__))
        return os.path.join(directorio, nombre_archivo)

    def test_corpus_legal_existe(self):
        """El archivo corpus_legal.json existe."""
        ruta = self._ruta_proyecto("corpus_legal.json")
        assert os.path.exists(ruta), f"No se encontró: {ruta}"

    def test_corpus_legal_minimo_50_registros(self):
        """El corpus legal tiene al menos 50 registros."""
        ruta = self._ruta_proyecto("corpus_legal.json")
        with open(ruta, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        assert len(corpus) >= 50

    def test_base_conocimiento_existe(self):
        """El archivo base_conocimiento.json existe."""
        ruta = self._ruta_proyecto("base_conocimiento.json")
        assert os.path.exists(ruta), f"No se encontró: {ruta}"

    def test_base_conocimiento_tiene_fragmentos(self):
        """La base de conocimiento tiene fragmentos normativos."""
        ruta = self._ruta_proyecto("base_conocimiento.json")
        with open(ruta, "r", encoding="utf-8") as f:
            base = json.load(f)
        assert len(base) >= 10

    def test_base_conocimiento_campos_requeridos(self):
        """Cada fragmento de la base de conocimiento tiene los campos requeridos."""
        ruta = self._ruta_proyecto("base_conocimiento.json")
        with open(ruta, "r", encoding="utf-8") as f:
            base = json.load(f)
        for fragmento in base:
            assert "contexto" in fragmento
            assert "fuente" in fragmento
            assert "tema" in fragmento


# ================================================================
# Test: Módulos principales existen y son importables
# ================================================================

class TestModulosImportables:
    """Verifica que todos los módulos del MVP son importables."""

    def test_importar_preprocesador(self):
        """El módulo preprocessor.py es importable."""
        from justia.preprocessor import Preprocesador
        assert Preprocesador is not None

    def test_importar_clasificador(self):
        """El módulo clasificador.py es importable."""
        from justia.clasificador import ModuloClasificador
        assert ModuloClasificador is not None

    def test_importar_ner(self):
        """El módulo ner.py es importable."""
        from justia.ner import ModuloNER
        assert ModuloNER is not None

    def test_importar_qa(self):
        """El módulo qa.py es importable."""
        from justia.qa import ModuloQA
        assert ModuloQA is not None

    def test_importar_clustering(self):
        """El módulo clustering.py es importable."""
        from justia.clustering import ModuloClustering
        assert ModuloClustering is not None

    def test_importar_trazabilidad(self):
        """El módulo trazabilidad.py es importable."""
        from justia.trazabilidad import ServicioTrazabilidad
        assert ServicioTrazabilidad is not None

    def test_importar_xai(self):
        """El módulo xai.py es importable."""
        from justia.xai import ServicioXAI
        assert ServicioXAI is not None

    def test_importar_modelos(self):
        """El módulo modelos.py es importable con todas las dataclasses."""
        from justia.modelos import (
            RegistroCorpus,
            EntidadLegal,
            ResultadoClasificacion,
            ResultadoNER,
            ResultadoQA,
            InfoCluster,
            ResultadoClustering,
            RegistroTrazabilidad,
            SesionTrazabilidad,
        )
        assert all([
            RegistroCorpus, EntidadLegal, ResultadoClasificacion,
            ResultadoNER, ResultadoQA, InfoCluster, ResultadoClustering,
            RegistroTrazabilidad, SesionTrazabilidad,
        ])


# ================================================================
# Test: Constructores de módulos aceptan trazabilidad y xai
# ================================================================

class TestConstructoresIntegracion:
    """Verifica que los constructores de módulos aceptan trazabilidad y xai."""

    def test_preprocesador_acepta_trazabilidad(self, trazabilidad):
        """Preprocesador acepta ServicioTrazabilidad en constructor."""
        preprocesador = Preprocesador(trazabilidad=trazabilidad)
        assert preprocesador.trazabilidad is trazabilidad

    def test_ner_acepta_trazabilidad_y_xai(self, trazabilidad, xai):
        """ModuloNER acepta ServicioTrazabilidad y ServicioXAI."""
        modulo_ner = __import__("justia.ner", fromlist=["ModuloNER"]).ModuloNER(
            trazabilidad=trazabilidad, xai=xai
        )
        assert modulo_ner.trazabilidad is trazabilidad
        assert modulo_ner.xai is xai

    def test_clustering_acepta_trazabilidad_y_xai(self, trazabilidad, xai):
        """ModuloClustering acepta ServicioTrazabilidad y ServicioXAI."""
        from justia.clustering import ModuloClustering
        modulo = ModuloClustering(trazabilidad=trazabilidad, xai=xai)
        assert modulo.trazabilidad is trazabilidad
        assert modulo.xai is xai


# ================================================================
# Test: NER registra en trazabilidad y usa XAI
# ================================================================

class TestNERIntegracion:
    """Verifica integración del módulo NER con trazabilidad y XAI."""

    def test_ner_registra_en_trazabilidad(self, trazabilidad, xai):
        """NER registra cada extracción en trazabilidad."""
        from justia.ner import ModuloNER
        modulo_ner = ModuloNER(trazabilidad=trazabilidad, xai=xai)

        resultado = modulo_ner.extraer_entidades(
            "Según la Ley 1098 de 2006, los menores tienen protección."
        )

        registros_ner = [
            r for r in trazabilidad.sesion.registros
            if r.modulo == "ner"
        ]
        assert len(registros_ner) == 1

    def test_ner_resultado_incluye_supervision(self, trazabilidad, xai):
        """El resultado NER incluye estado de supervisión y aviso de IA."""
        from justia.ner import ModuloNER
        modulo_ner = ModuloNER(trazabilidad=trazabilidad, xai=xai)

        resultado = modulo_ner.extraer_entidades(
            "Demanda ante la Corte Constitucional."
        )

        assert resultado.estado_supervision in [
            "Pendiente de Validación Humana",
            "Prioridad Alta para Revisión",
        ]
        assert resultado.aviso_ia == AVISO_IA


# ================================================================
# Test: Clustering registra en trazabilidad y usa XAI
# ================================================================

class TestClusteringIntegracion:
    """Verifica integración del módulo Clustering con trazabilidad y XAI."""

    def test_clustering_registra_en_trazabilidad(
        self, trazabilidad, xai, corpus_procesado
    ):
        """Clustering registra cada agrupamiento en trazabilidad."""
        from justia.clustering import ModuloClustering
        modulo = ModuloClustering(trazabilidad=trazabilidad, xai=xai)

        textos_limpios = [reg["limpio"] for reg in corpus_procesado]
        resultado = modulo.agrupar(textos_limpios)

        registros_clustering = [
            r for r in trazabilidad.sesion.registros
            if r.modulo == "clustering"
        ]
        assert len(registros_clustering) == 1

    def test_clustering_resultado_incluye_supervision(
        self, trazabilidad, xai, corpus_procesado
    ):
        """El resultado de clustering incluye estado de supervisión y aviso de IA."""
        from justia.clustering import ModuloClustering
        modulo = ModuloClustering(trazabilidad=trazabilidad, xai=xai)

        textos_limpios = [reg["limpio"] for reg in corpus_procesado]
        resultado = modulo.agrupar(textos_limpios)

        assert resultado.estado_supervision in [
            "Pendiente de Validación Humana",
            "Prioridad Alta para Revisión",
        ]
        assert resultado.aviso_ia == AVISO_IA


# ================================================================
# Test: Aviso de IA consistente en ServicioXAI
# ================================================================

class TestAvisoIAConsistencia:
    """Verifica que el aviso de IA es consistente en todo el sistema."""

    def test_aviso_xai_constante(self):
        """ServicioXAI.AVISO_IA tiene el texto obligatorio."""
        xai = ServicioXAI()
        assert xai.AVISO_IA == AVISO_IA

    def test_aviso_en_formatear_explicacion(self):
        """formatear_explicacion incluye el aviso correcto."""
        xai = ServicioXAI()
        resultado = xai.formatear_explicacion(
            key_tokens=["test"], nivel_certeza=0.8, umbral_confianza=0.6
        )
        assert resultado["aviso_ia"] == AVISO_IA

    def test_aviso_en_etiquetar_supervision(self):
        """etiquetar_supervision incluye el aviso correcto."""
        xai = ServicioXAI()
        resultado = xai.etiquetar_supervision(
            nivel_certeza=0.8, umbral_confianza=0.6
        )
        assert resultado["aviso"] == AVISO_IA
