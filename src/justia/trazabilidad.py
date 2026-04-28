"""
Servicio de Trazabilidad del sistema JustIA MVP.

Proporciona registro centralizado de operaciones para auditoría,
incluyendo anonimización de datos personales y exportación a JSON.
Cada operación de cada módulo se registra con marca temporal ISO 8601,
versión del modelo, datos de entrada anonimizados y resultado producido.

Valida: Requisitos 7.1, 7.2, 7.3, 7.4, 10.2
"""

import json
import re
from dataclasses import asdict
from datetime import datetime

from justia.modelos import RegistroTrazabilidad, SesionTrazabilidad


class ServicioTrazabilidad:
    """Registro centralizado de operaciones para auditoría.

    Mantiene una sesión de trazabilidad en memoria que agrupa todos los
    registros de una ejecución del MVP, incluyendo versiones de modelos
    y dependencias del entorno para garantizar reproducibilidad.

    Attributes:
        sesion: Sesión de trazabilidad con registros acumulados.
    """

    # Patrones regex para anonimización de datos personales colombianos
    # Nombres propios: secuencias de 2-4 palabras capitalizadas (e.g. "Juan Carlos Pérez")
    _PATRON_NOMBRE = re.compile(
        r"\b(?:[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+\s+)"
        r"(?:(?:de(?:l)?|la|los|las)\s+)?"
        r"(?:[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+\s*){1,3}\b"
    )

    # Cédulas colombianas: secuencias de 8 a 10 dígitos, posiblemente con puntos
    _PATRON_CEDULA = re.compile(
        r"\b\d{1,3}(?:\.\d{3}){2,3}\b"  # formato con puntos: 1.234.567.890
        r"|\b\d{8,10}\b"                  # formato sin puntos: 1234567890
    )

    # Teléfonos colombianos: formato 3XX XXXXXXX o variantes con separadores
    _PATRON_TELEFONO = re.compile(
        r"\b3\d{2}[\s\-]?\d{3}[\s\-]?\d{4}\b"       # celular: 3XX XXX XXXX
        r"|\b\(\d{1,3}\)\s?\d{7}\b"                    # fijo: (1) 1234567
        r"|\b\+57\s?\d{10}\b"                          # internacional: +57 3001234567
    )

    # Direcciones colombianas: patrones con Calle, Carrera, Avenida, etc.
    _PATRON_DIRECCION = re.compile(
        r"\b(?:Calle|Carrera|Avenida|Av\.|Cra\.|Cl\.|Diagonal|Transversal|Kra\.)"
        r"\s+\d+[A-Za-z]?(?:\s*(?:#|No\.?|Nro\.?)\s*\d+[A-Za-z]?\s*-?\s*\d*)?",
        re.IGNORECASE,
    )

    def __init__(self):
        """Inicializa el almacén de registros en memoria.

        Crea una SesionTrazabilidad con la marca temporal actual en
        formato ISO 8601 y listas vacías para registros, versiones
        de modelos y dependencias del entorno.
        """
        self.sesion = SesionTrazabilidad(
            inicio_sesion=datetime.now().isoformat(),
            versiones_modelos={},
            dependencias_entorno={},
            registros=[],
        )

    def registrar_inicio_sesion(
        self, versiones_modelos: dict, dependencias: dict
    ) -> None:
        """Registra el inicio de sesión con versiones de modelos y dependencias.

        Args:
            versiones_modelos: Diccionario con las versiones de cada modelo
                cargado (e.g. {"clasificador": "mDeBERTa-v3-base-mnli-xnli"}).
            dependencias: Diccionario con las versiones de las dependencias
                del entorno (e.g. {"python": "3.10", "torch": "2.1.0"}).
        """
        self.sesion.versiones_modelos = versiones_modelos
        self.sesion.dependencias_entorno = dependencias

    def registrar_operacion(
        self,
        modulo: str,
        entrada: str,
        resultado: dict,
        version_modelo: str,
        metadata: dict = None,
    ) -> None:
        """Registra una operación individual de cualquier módulo.

        Crea un RegistroTrazabilidad con timestamp ISO 8601 y anonimiza
        los datos de entrada antes de almacenarlos para proteger la
        privacidad del usuario consultante.

        Args:
            modulo: Nombre del módulo que ejecutó la operación
                (e.g. "clasificador", "ner", "qa", "clustering").
            entrada: Texto de entrada de la operación (será anonimizado).
            resultado: Diccionario con el resultado de la operación.
            version_modelo: Versión o identificador del modelo utilizado.
            metadata: Información adicional de la operación (opcional).
        """
        registro = RegistroTrazabilidad(
            timestamp=datetime.now().isoformat(),
            modulo=modulo,
            version_modelo=version_modelo,
            entrada_anonimizada=self._anonimizar(entrada),
            resultado=resultado,
            metadata=metadata if metadata is not None else {},
        )
        self.sesion.registros.append(registro)

    def exportar_registros(
        self, ruta: str = "registros_trazabilidad.json"
    ) -> None:
        """Exporta todos los registros a archivo JSON estructurado.

        Serializa la sesión completa de trazabilidad a un archivo JSON
        legible, incluyendo información de sesión (versiones, dependencias)
        y todos los registros de operaciones.

        Args:
            ruta: Ruta del archivo de salida. Por defecto
                "registros_trazabilidad.json".
        """
        datos = {
            "sesion": {
                "inicio": self.sesion.inicio_sesion,
                "versiones_modelos": self.sesion.versiones_modelos,
                "dependencias": self.sesion.dependencias_entorno,
            },
            "registros": [asdict(r) for r in self.sesion.registros],
        }
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(datos, f, ensure_ascii=False, indent=2)

    def _anonimizar(self, texto: str) -> str:
        """Anonimiza datos personales en el texto usando patrones regex.

        Reemplaza patrones de datos personales colombianos con marcadores
        genéricos para cumplir con los principios de privacidad por diseño.

        Orden de aplicación:
        1. Direcciones -> [DIRECCION]
        2. Teléfonos -> [TELEFONO]
        3. Cédulas -> [CEDULA]
        4. Nombres propios -> [PERSONA]

        El orden importa: las direcciones y teléfonos se procesan primero
        para evitar que sus componentes numéricos sean capturados por el
        patrón de cédulas, y los nombres se procesan al final para no
        interferir con los patrones de direcciones.

        Args:
            texto: Texto con posibles datos personales.

        Returns:
            Texto con datos personales reemplazados por marcadores genéricos.
        """
        # 1. Direcciones (antes de nombres para no capturar "Calle" como nombre)
        texto = self._PATRON_DIRECCION.sub("[DIRECCION]", texto)

        # 2. Teléfonos (antes de cédulas para no confundir formatos numéricos)
        texto = self._PATRON_TELEFONO.sub("[TELEFONO]", texto)

        # 3. Cédulas (números de 8-10 dígitos)
        texto = self._PATRON_CEDULA.sub("[CEDULA]", texto)

        # 4. Nombres propios (secuencias de palabras capitalizadas)
        texto = self._PATRON_NOMBRE.sub("[PERSONA]", texto)

        return texto
