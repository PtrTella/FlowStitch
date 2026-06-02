"""
Decoder sperimentali per l'estrazione di maschere ibride.

Percorso di ricerca (Capitolo 5):
1. SupervisedHybridDecoder      — cosine similarity topologica + sigmoid gating
2. SvdCosineHybridDecoder       — SVD sulle 24 teste per consenso + cosine topology
3. SupervisedSpectralHybridDecoder — vettore di Fiedler dal Laplaciano normalizzato
4. GraphDiffusionHybridDecoder   — random walk di Markov sul grafo coseno
5. SvdSeededGraphDiffusionDecoder — SVD seed + diffusione di Markov (vincitore)
"""
from .supervised_hybrid import SupervisedHybridDecoder
from .svd_cosine_hybrid import SvdCosineHybridDecoder
from .spectral_hybrid import SupervisedSpectralHybridDecoder
from .graph_diffusion import GraphDiffusionHybridDecoder
from .svd_graph_diffusion import SvdSeededGraphDiffusionDecoder

__all__ = [
    "SupervisedHybridDecoder",
    "SvdCosineHybridDecoder",
    "SupervisedSpectralHybridDecoder",
    "GraphDiffusionHybridDecoder",
    "SvdSeededGraphDiffusionDecoder",
]
