"""
Tests unitarios para el Servicio XAI del sistema JustIA MVP.

Verifica el correcto funcionamiento de formatear_explicacion y
etiquetar_supervision, incluyendo:
- Formato de certeza como porcentaje (Req 6.2)
- Key tokens en orden descendente de relevancia (Req 6.3)
- Flag de revisión cuando certeza < umbral (Req 6.4)
- Aviso de IA obligatorio en todas las salidas (Req 8.2)
- Estado "Pendiente de Validación Humana" por defecto (Req 8.1)
- Escalamiento a "Prioridad Alta para Revisión" con baja confianza (Req 8.3)
- Factores de incertidumbre cuando certeza < umbral (Req 8.3)

Valida: Requisitos 6.1, 6.2, 6.3, 6.4, 8.1, 8.2, 8.3
"""

from justia.xai import ServicioXAI


class TestFormatearExplicacion:
    """Tests para el método formatear_explicacion."""

    def setup_method(self):
        """Inicializa el servicio XAI antes de cada test."""
        self.xai = ServicioXAI()

    def test_certeza_como_porcentaje(self):
        """Req 6.2: El nivel de certeza se muestra como porcentaje."""
        resultado = self.xai.formatear_explicacion(
            key_tokens=["divorcio", "matrimonial"],
            nivel_certeza=0.853,
            umbral_confianza=0.6,
        )
        assert resultado["certeza_porcentaje"] == "85.3%"

    def test_certeza_cero_porcentaje(self):
        """Certeza 0.0 se formatea como '0.0%'."""
        resultado = self.xai.formatear_explicacion(
            key_tokens=["token"],
            nivel_certeza=0.0,
            umbral_confianza=0.6,
        )
        assert resultado["certeza_porcentaje"] == "0.0%"

    def test_certeza_uno_porcentaje(self):
        """Certeza 1.0 se formatea como '100.0%'."""
        resultado = self.xai.formatear_explicacion(
            key_tokens=["token"],
            nivel_certeza=1.0,
            umbral_confianza=0.6,
        )
        assert resultado["certeza_porcentaje"] == "100.0%"

    def test_key_tokens_preservados_en_orden(self):
        """Req 6.3: Key tokens se presentan en orden descendente de relevancia."""
        tokens = ["custodia", "menor", "familia", "protección"]
        resultado = self.xai.formatear_explicacion(
            key_tokens=tokens,
            nivel_certeza=0.8,
            umbral_confianza=0.6,
        )
        assert resultado["key_tokens"] == tokens

    def test_key_tokens_son_copia(self):
        """Los key_tokens retornados son una copia, no la lista original."""
        tokens_originales = ["divorcio", "matrimonial"]
        resultado = self.xai.formatear_explicacion(
            key_tokens=tokens_originales,
            nivel_certeza=0.8,
            umbral_confianza=0.6,
        )
        resultado["key_tokens"].append("extra")
        assert len(tokens_originales) == 2

    def test_requiere_revision_cuando_certeza_bajo_umbral(self):
        """Req 6.4: Flag de revisión True cuando certeza < umbral."""
        resultado = self.xai.formatear_explicacion(
            key_tokens=["token"],
            nivel_certeza=0.4,
            umbral_confianza=0.6,
        )
        assert resultado["requiere_revision"] is True

    def test_no_requiere_revision_cuando_certeza_sobre_umbral(self):
        """Flag de revisión False cuando certeza >= umbral."""
        resultado = self.xai.formatear_explicacion(
            key_tokens=["token"],
            nivel_certeza=0.8,
            umbral_confianza=0.6,
        )
        assert resultado["requiere_revision"] is False

    def test_no_requiere_revision_cuando_certeza_igual_umbral(self):
        """Flag de revisión False cuando certeza == umbral."""
        resultado = self.xai.formatear_explicacion(
            key_tokens=["token"],
            nivel_certeza=0.6,
            umbral_confianza=0.6,
        )
        assert resultado["requiere_revision"] is False

    def test_aviso_ia_presente(self):
        """Req 8.2: Aviso de IA obligatorio incluido en la explicación."""
        resultado = self.xai.formatear_explicacion(
            key_tokens=["token"],
            nivel_certeza=0.8,
            umbral_confianza=0.6,
        )
        assert resultado["aviso_ia"] == (
            "Este resultado fue generado por IA y requiere validación "
            "de un profesional del derecho antes de ser utilizado."
        )

    def test_estructura_completa_del_dict(self):
        """La explicación contiene todas las claves requeridas."""
        resultado = self.xai.formatear_explicacion(
            key_tokens=["divorcio"],
            nivel_certeza=0.75,
            umbral_confianza=0.6,
        )
        assert set(resultado.keys()) == {
            "key_tokens",
            "certeza_porcentaje",
            "requiere_revision",
            "aviso_ia",
        }


class TestEtiquetarSupervision:
    """Tests para el método etiquetar_supervision."""

    def setup_method(self):
        """Inicializa el servicio XAI antes de cada test."""
        self.xai = ServicioXAI()

    def test_estado_pendiente_cuando_certeza_sobre_umbral(self):
        """Req 8.1: Estado 'Pendiente de Validación Humana' cuando certeza >= umbral."""
        resultado = self.xai.etiquetar_supervision(
            nivel_certeza=0.8,
            umbral_confianza=0.6,
        )
        assert resultado["estado"] == "Pendiente de Validación Humana"

    def test_estado_prioridad_alta_cuando_certeza_bajo_umbral(self):
        """Req 8.3: Estado 'Prioridad Alta para Revisión' cuando certeza < umbral."""
        resultado = self.xai.etiquetar_supervision(
            nivel_certeza=0.4,
            umbral_confianza=0.6,
        )
        assert resultado["estado"] == "Prioridad Alta para Revisión"

    def test_estado_pendiente_cuando_certeza_igual_umbral(self):
        """Estado 'Pendiente de Validación Humana' cuando certeza == umbral."""
        resultado = self.xai.etiquetar_supervision(
            nivel_certeza=0.6,
            umbral_confianza=0.6,
        )
        assert resultado["estado"] == "Pendiente de Validación Humana"

    def test_aviso_presente(self):
        """Req 8.2: Aviso de IA obligatorio incluido en la etiqueta."""
        resultado = self.xai.etiquetar_supervision(
            nivel_certeza=0.8,
            umbral_confianza=0.6,
        )
        assert resultado["aviso"] == (
            "Este resultado fue generado por IA y requiere validación "
            "de un profesional del derecho antes de ser utilizado."
        )

    def test_factores_incertidumbre_vacios_cuando_certeza_alta(self):
        """Factores de incertidumbre vacíos cuando certeza >= umbral."""
        resultado = self.xai.etiquetar_supervision(
            nivel_certeza=0.8,
            umbral_confianza=0.6,
        )
        assert resultado["factores_incertidumbre"] == []

    def test_factores_incertidumbre_presentes_cuando_certeza_baja(self):
        """Req 8.3: Factores de incertidumbre presentes cuando certeza < umbral."""
        resultado = self.xai.etiquetar_supervision(
            nivel_certeza=0.3,
            umbral_confianza=0.6,
        )
        factores = resultado["factores_incertidumbre"]
        assert len(factores) > 0
        # Verificar que los factores mencionan el nivel de certeza
        assert any("30.0%" in f for f in factores)
        # Verificar que los factores mencionan el umbral
        assert any("60.0%" in f for f in factores)

    def test_estructura_completa_del_dict(self):
        """La etiqueta de supervisión contiene todas las claves requeridas."""
        resultado = self.xai.etiquetar_supervision(
            nivel_certeza=0.75,
            umbral_confianza=0.6,
        )
        assert set(resultado.keys()) == {
            "estado",
            "aviso",
            "factores_incertidumbre",
        }

    def test_certeza_cero_es_prioridad_alta(self):
        """Certeza 0.0 siempre genera 'Prioridad Alta para Revisión'."""
        resultado = self.xai.etiquetar_supervision(
            nivel_certeza=0.0,
            umbral_confianza=0.6,
        )
        assert resultado["estado"] == "Prioridad Alta para Revisión"
        assert len(resultado["factores_incertidumbre"]) > 0
