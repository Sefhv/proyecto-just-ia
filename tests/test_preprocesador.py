"""
Tests unitarios para el Preprocesador de textos legales.

Verifica la normalización de textos, el manejo de textos inválidos,
el procesamiento de corpus y la integración con trazabilidad.

Valida: Requisitos 1.1, 1.2, 1.3, 1.4
"""

import pytest

from justia.preprocessor import Preprocesador
from justia.trazabilidad import ServicioTrazabilidad


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def preprocesador():
    """Instancia de Preprocesador sin trazabilidad."""
    return Preprocesador()


@pytest.fixture
def preprocesador_con_trazabilidad():
    """Instancia de Preprocesador con ServicioTrazabilidad."""
    traz = ServicioTrazabilidad()
    return Preprocesador(trazabilidad=traz), traz


# ---------------------------------------------------------------------------
# Tests de normalizar_texto (Requisito 1.1)
# ---------------------------------------------------------------------------

class TestNormalizarTexto:
    """Tests para el método normalizar_texto."""

    def test_conversion_a_minusculas(self, preprocesador):
        """El texto normalizado debe estar en minúsculas."""
        resultado = preprocesador.normalizar_texto("DEMANDA DE DIVORCIO")
        assert resultado == resultado.lower()

    def test_eliminacion_caracteres_especiales(self, preprocesador):
        """Los caracteres especiales deben eliminarse, conservando alfanuméricos y espacios."""
        resultado = preprocesador.normalizar_texto("Ley 1098 de 2006!")
        # No debe contener signos de puntuación
        assert "!" not in resultado
        assert all(c.isalnum() or c.isspace() for c in resultado)

    def test_lematizacion_aplicada(self, preprocesador):
        """Las palabras deben estar lematizadas."""
        resultado = preprocesador.normalizar_texto("Los trabajadores reclamaron sus derechos")
        # spaCy debería lematizar "trabajadores" -> "trabajador" o similar
        # y "reclamaron" -> "reclamar" o similar
        palabras = resultado.split()
        assert len(palabras) > 0
        # Verificar que el resultado es diferente del input en minúsculas sin especiales
        assert resultado != "los trabajadores reclamaron sus derechos"

    def test_texto_legal_colombiano(self, preprocesador):
        """Debe normalizar correctamente un texto legal colombiano típico."""
        texto = "La Ley 1098 de 2006 establece protección para menores."
        resultado = preprocesador.normalizar_texto(texto)
        assert "ley" in resultado or "el" in resultado
        assert "1098" in resultado
        assert "2006" in resultado
        assert "." not in resultado

    def test_espacios_multiples_colapsados(self, preprocesador):
        """Los espacios múltiples deben colapsarse a uno solo."""
        resultado = preprocesador.normalizar_texto("demanda   de    divorcio")
        assert "  " not in resultado


# ---------------------------------------------------------------------------
# Tests de ValueError (Requisito 1.4)
# ---------------------------------------------------------------------------

class TestTextoInvalido:
    """Tests para el rechazo de textos inválidos."""

    def test_texto_vacio_lanza_valueerror(self, preprocesador):
        """Un texto vacío debe lanzar ValueError."""
        with pytest.raises(ValueError):
            preprocesador.normalizar_texto("")

    def test_solo_caracteres_especiales_lanza_valueerror(self, preprocesador):
        """Un texto con solo caracteres especiales debe lanzar ValueError."""
        with pytest.raises(ValueError):
            preprocesador.normalizar_texto("!!!@@@###$$$")

    def test_solo_espacios_lanza_valueerror(self, preprocesador):
        """Un texto con solo espacios debe lanzar ValueError."""
        with pytest.raises(ValueError):
            preprocesador.normalizar_texto("     ")

    def test_espacios_y_especiales_lanza_valueerror(self, preprocesador):
        """Un texto con solo espacios y caracteres especiales debe lanzar ValueError."""
        with pytest.raises(ValueError):
            preprocesador.normalizar_texto("  !!! ### $$$ ")


# ---------------------------------------------------------------------------
# Tests de procesar_corpus (Requisitos 1.2, 1.3, 1.4)
# ---------------------------------------------------------------------------

class TestProcesarCorpus:
    """Tests para el método procesar_corpus."""

    def test_formato_campos_original_y_limpio(self, preprocesador):
        """Cada registro procesado debe tener campos 'original' y 'limpio' (Req 1.2)."""
        corpus = [{"texto": "Demanda de divorcio por abandono."}]
        resultado = preprocesador.procesar_corpus(corpus)
        assert len(resultado) == 1
        assert "original" in resultado[0]
        assert "limpio" in resultado[0]

    def test_campo_original_preservado(self, preprocesador):
        """El campo 'original' debe ser idéntico al texto de entrada (Req 1.2)."""
        texto = "Demanda de divorcio por abandono."
        corpus = [{"texto": texto}]
        resultado = preprocesador.procesar_corpus(corpus)
        assert resultado[0]["original"] == texto

    def test_campo_limpio_no_vacio(self, preprocesador):
        """El campo 'limpio' debe ser una cadena no vacía (Req 1.2)."""
        corpus = [{"texto": "Demanda de divorcio por abandono."}]
        resultado = preprocesador.procesar_corpus(corpus)
        assert len(resultado[0]["limpio"]) > 0

    def test_registros_invalidos_excluidos(self, preprocesador):
        """Los registros con textos inválidos deben excluirse (Req 1.4)."""
        corpus = [
            {"texto": "Texto válido de prueba."},
            {"texto": ""},
            {"texto": "!!!###"},
        ]
        resultado = preprocesador.procesar_corpus(corpus)
        assert len(resultado) == 1

    def test_corpus_minimo_50_registros(self, preprocesador, corpus_minimo):
        """El corpus procesado debe tener al menos 50 registros (Req 1.3)."""
        resultado = preprocesador.procesar_corpus(corpus_minimo)
        assert len(resultado) >= 50

    def test_corpus_vacio_retorna_lista_vacia(self, preprocesador):
        """Un corpus vacío debe retornar una lista vacía."""
        resultado = preprocesador.procesar_corpus([])
        assert resultado == []


# ---------------------------------------------------------------------------
# Tests de integración con trazabilidad (Requisito 1.4)
# ---------------------------------------------------------------------------

class TestIntegracionTrazabilidad:
    """Tests para la integración con ServicioTrazabilidad."""

    def test_texto_invalido_registra_advertencia(self, preprocesador_con_trazabilidad):
        """Un texto inválido debe registrar advertencia en trazabilidad (Req 1.4)."""
        prep, traz = preprocesador_con_trazabilidad
        corpus = [{"texto": ""}]
        prep.procesar_corpus(corpus)
        assert len(traz.sesion.registros) == 1
        assert traz.sesion.registros[0].modulo == "preprocesador"
        assert traz.sesion.registros[0].resultado["estado"] == "excluido"

    def test_multiples_invalidos_registran_advertencias(self, preprocesador_con_trazabilidad):
        """Múltiples textos inválidos deben registrar múltiples advertencias."""
        prep, traz = preprocesador_con_trazabilidad
        corpus = [
            {"texto": ""},
            {"texto": "!!!"},
            {"texto": "   "},
        ]
        prep.procesar_corpus(corpus)
        assert len(traz.sesion.registros) == 3

    def test_texto_valido_no_registra_advertencia(self, preprocesador_con_trazabilidad):
        """Un texto válido no debe registrar advertencia en trazabilidad."""
        prep, traz = preprocesador_con_trazabilidad
        corpus = [{"texto": "Demanda de divorcio por abandono."}]
        prep.procesar_corpus(corpus)
        assert len(traz.sesion.registros) == 0

    def test_sin_trazabilidad_no_falla(self, preprocesador):
        """El preprocesador sin trazabilidad no debe fallar con textos inválidos."""
        corpus = [{"texto": ""}, {"texto": "Texto válido."}]
        resultado = preprocesador.procesar_corpus(corpus)
        assert len(resultado) == 1
