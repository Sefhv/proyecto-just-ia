"""
Módulo de Agrupamiento Temático de casos legales del sistema JustIA MVP.

Agrupa casos legales por similitud temática utilizando TF-IDF para
vectorización y K-Means para agrupamiento. Cada cluster incluye las
tres palabras clave más representativas y un resumen descriptivo del
tema. Integra supervisión humana obligatoria y trazabilidad de cada
operación de agrupamiento.

Valida: Requisitos 5.1, 5.2, 5.3, 5.4, 5.5
"""

import logging
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from justia.modelos import InfoCluster, ResultadoClustering
from justia.trazabilidad import ServicioTrazabilidad
from justia.xai import ServicioXAI

logger = logging.getLogger(__name__)

# Mínimo de textos requeridos para ejecutar el agrupamiento
_MIN_TEXTOS = 50


class ModuloClustering:
    """Agrupamiento temático de casos legales.

    Utiliza TF-IDF para vectorizar textos legales y K-Means para
    agruparlos en clusters temáticos. Cada cluster reporta las tres
    palabras clave más representativas (extraídas de los pesos TF-IDF
    del centroide) y un resumen descriptivo del tema.

    Attributes:
        n_clusters: Número de clusters a generar.
        max_features: Número máximo de features para TF-IDF.
        vectorizer: Instancia de TfidfVectorizer configurada.
        kmeans: Instancia de KMeans configurada.
        xai: Servicio de explicabilidad para etiquetas de supervisión.
        trazabilidad: Servicio de trazabilidad para registrar operaciones.
    """

    def __init__(
        self,
        n_clusters: int = 4,
        max_features: int = 1000,
        trazabilidad: Optional[ServicioTrazabilidad] = None,
        xai: Optional[ServicioXAI] = None,
    ):
        """Configura TF-IDF vectorizer y K-Means.

        Inicializa el vectorizador TF-IDF con stop words en español y
        el algoritmo K-Means con semilla fija para reproducibilidad.

        Args:
            n_clusters: Número de clusters a generar. Por defecto 4.
            max_features: Número máximo de features para el vectorizador
                TF-IDF. Por defecto 1000.
            trazabilidad: Instancia opcional de ServicioTrazabilidad para
                registrar cada agrupamiento realizado.
            xai: Instancia opcional de ServicioXAI para generar etiquetas
                de supervisión humana.
        """
        self.n_clusters = n_clusters
        self.max_features = max_features
        self.trazabilidad = trazabilidad
        self.xai = xai if xai is not None else ServicioXAI()

        # Vectorizador TF-IDF con stop words en español
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english",  # sklearn no tiene stop words en español integradas
        )

        # K-Means con semilla fija para reproducibilidad
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10,
        )

    def agrupar(self, textos: list[str]) -> ResultadoClustering:
        """Agrupa textos por similitud temática.

        Vectoriza los textos con TF-IDF y los agrupa con K-Means.
        Para cada cluster genera las tres palabras clave más
        representativas y un resumen descriptivo del tema.

        Args:
            textos: Lista de textos legales preprocesados. Debe contener
                al menos 50 textos.

        Returns:
            ResultadoClustering con la lista de clusters, número de
            clusters, total de casos, inercia y estado de supervisión.

        Raises:
            ValueError: Si hay menos de 50 textos en la lista de entrada.
        """
        # Validar mínimo de textos
        if len(textos) < _MIN_TEXTOS:
            raise ValueError(
                f"Se requieren al menos {_MIN_TEXTOS} textos para el "
                f"agrupamiento. Se recibieron {len(textos)}."
            )

        # Vectorizar textos con TF-IDF
        tfidf_matrix = self.vectorizer.fit_transform(textos)

        # Ajustar K-Means sobre la matriz TF-IDF
        self.kmeans.fit(tfidf_matrix)

        # Obtener etiquetas de cluster asignadas a cada texto
        labels = self.kmeans.labels_

        # Construir información de cada cluster
        clusters: list[InfoCluster] = []
        for cluster_id in range(self.n_clusters):
            # Contar casos asignados a este cluster
            num_casos = int(np.sum(labels == cluster_id))

            # Obtener las 3 palabras clave más representativas
            palabras_clave = self._obtener_palabras_clave(cluster_id, n_palabras=3)

            # Generar resumen descriptivo del tema
            resumen = self._generar_resumen(cluster_id, palabras_clave)

            clusters.append(
                InfoCluster(
                    cluster_id=cluster_id,
                    num_casos=num_casos,
                    palabras_clave=palabras_clave,
                    resumen=resumen,
                )
            )

        # Obtener inercia (suma de distancias al centroide)
        inercia = float(self.kmeans.inertia_)

        # Generar etiquetas de supervisión humana
        supervision = self.xai.etiquetar_supervision(
            nivel_certeza=1.0,  # Clustering no tiene certeza individual
            umbral_confianza=0.5,
        )

        # Construir resultado de clustering
        resultado = ResultadoClustering(
            clusters=clusters,
            num_clusters=self.n_clusters,
            num_casos_total=len(textos),
            inercia=inercia,
            estado_supervision=supervision["estado"],
            aviso_ia=supervision["aviso"],
        )

        # Registrar en trazabilidad
        if self.trazabilidad is not None:
            self.trazabilidad.registrar_operacion(
                modulo="clustering",
                entrada=f"{len(textos)} textos procesados",
                resultado={
                    "num_clusters": self.n_clusters,
                    "num_casos_total": len(textos),
                    "inercia": inercia,
                    "clusters": [
                        {
                            "cluster_id": c.cluster_id,
                            "num_casos": c.num_casos,
                            "palabras_clave": c.palabras_clave,
                        }
                        for c in clusters
                    ],
                },
                version_modelo="sklearn TF-IDF + KMeans",
                metadata={
                    "n_clusters": self.n_clusters,
                    "max_features": self.max_features,
                },
            )

        return resultado

    def _obtener_palabras_clave(
        self, cluster_id: int, n_palabras: int = 3
    ) -> list[str]:
        """Obtiene las N palabras más representativas de un cluster.

        Extrae las palabras con mayor peso TF-IDF en el centroide del
        cluster indicado. Los centroides de K-Means representan el
        "documento promedio" de cada cluster, por lo que las features
        con mayor peso son las más representativas del tema.

        Args:
            cluster_id: Identificador numérico del cluster.
            n_palabras: Número de palabras clave a extraer. Por defecto 3.

        Returns:
            Lista de las N palabras más representativas del cluster,
            ordenadas por peso TF-IDF descendente.
        """
        # Obtener el centroide del cluster
        centroide = self.kmeans.cluster_centers_[cluster_id]

        # Obtener los nombres de las features (palabras) del vectorizador
        feature_names = self.vectorizer.get_feature_names_out()

        # Encontrar los índices de las N features con mayor peso
        top_indices = np.argsort(centroide)[::-1][:n_palabras]

        # Retornar las palabras correspondientes
        return [str(feature_names[i]) for i in top_indices]

    def _generar_resumen(
        self, cluster_id: int, palabras_clave: list[str]
    ) -> str:
        """Genera un resumen descriptivo del tema del cluster.

        Crea una descripción legible del tema del cluster basada en
        sus palabras clave más representativas.

        Args:
            cluster_id: Identificador numérico del cluster.
            palabras_clave: Lista de palabras clave del cluster.

        Returns:
            Resumen descriptivo del tema del cluster.
        """
        keywords_str = ", ".join(palabras_clave)
        return f"Grupo temático relacionado con: {keywords_str}"
