"""
Servicio de Explicabilidad (XAI) del sistema JustIA MVP.

Proporciona transparencia y explicabilidad para las predicciones de IA,
formateando explicaciones con key tokens, niveles de certeza y etiquetas
de supervisión humana obligatoria. Cada salida incluye un aviso visible
de que el resultado fue generado por IA y requiere validación profesional.

Valida: Requisitos 6.1, 6.2, 6.3, 6.4, 8.1, 8.2, 8.3
"""


class ServicioXAI:
    """Transparencia y explicabilidad para predicciones de IA.

    Servicio transversal que formatea las explicaciones XAI para todos
    los módulos del sistema JustIA. Garantiza que cada predicción incluya
    key tokens ordenados por relevancia, nivel de certeza como porcentaje,
    flag de revisión humana y aviso obligatorio de IA.

    El aviso de IA es constante y se incluye en todas las salidas:
    "Este resultado fue generado por IA y requiere validación de un
    profesional del derecho antes de ser utilizado."
    """

    # Aviso obligatorio incluido en todas las salidas del sistema
    AVISO_IA = (
        "Este resultado fue generado por IA y requiere validación "
        "de un profesional del derecho antes de ser utilizado."
    )

    def formatear_explicacion(
        self,
        key_tokens: list[str],
        nivel_certeza: float,
        umbral_confianza: float,
    ) -> dict:
        """Formatea la explicación XAI para una predicción.

        Genera un diccionario con los elementos de explicabilidad requeridos:
        key tokens ordenados por relevancia descendente, certeza como
        porcentaje, flag de revisión humana y aviso obligatorio de IA.

        Los key_tokens se reciben ya ordenados por relevancia descendente
        desde el módulo que los genera. Este método preserva ese orden
        para garantizar que el supervisor humano vea primero los tokens
        más influyentes en la decisión del modelo.

        Args:
            key_tokens: Tokens ordenados por relevancia descendente.
            nivel_certeza: Score de confianza del modelo, rango [0, 1].
            umbral_confianza: Umbral mínimo configurado para considerar
                una predicción como confiable.

        Returns:
            Dict con las claves:
                - key_tokens: lista de tokens en orden descendente de relevancia.
                - certeza_porcentaje: nivel de certeza formateado como "XX.X%".
                - requiere_revision: True si nivel_certeza < umbral_confianza.
                - aviso_ia: aviso obligatorio de generación por IA.
        """
        return {
            "key_tokens": list(key_tokens),
            "certeza_porcentaje": f"{nivel_certeza * 100:.1f}%",
            "requiere_revision": nivel_certeza < umbral_confianza,
            "aviso_ia": self.AVISO_IA,
        }

    def etiquetar_supervision(
        self,
        nivel_certeza: float,
        umbral_confianza: float,
    ) -> dict:
        """Genera etiquetas de supervisión humana obligatoria.

        Toda salida del sistema requiere validación humana. Cuando la
        certeza del modelo es inferior al umbral de confianza, el estado
        se escala a "Prioridad Alta para Revisión" y se incluyen los
        factores de incertidumbre para que el supervisor humano pueda
        evaluar la fiabilidad de la predicción.

        Args:
            nivel_certeza: Score de confianza del modelo, rango [0, 1].
            umbral_confianza: Umbral mínimo configurado para considerar
                una predicción como confiable.

        Returns:
            Dict con las claves:
                - estado: "Pendiente de Validación Humana" cuando la certeza
                    es >= umbral, o "Prioridad Alta para Revisión" cuando
                    la certeza es < umbral.
                - aviso: aviso obligatorio de generación por IA.
                - factores_incertidumbre: lista de factores de incertidumbre
                    cuando la certeza es baja (lista vacía si certeza >= umbral).
        """
        baja_confianza = nivel_certeza < umbral_confianza

        if baja_confianza:
            estado = "Prioridad Alta para Revisión"
            factores = [
                f"Nivel de certeza ({nivel_certeza * 100:.1f}%) inferior "
                f"al umbral configurado ({umbral_confianza * 100:.1f}%)",
                "Se recomienda revisión exhaustiva por profesional del derecho",
            ]
        else:
            estado = "Pendiente de Validación Humana"
            factores = []

        return {
            "estado": estado,
            "aviso": self.AVISO_IA,
            "factores_incertidumbre": factores,
        }
