"""celltype-agent: automated cell type annotation for single-cell & spatial genomics."""

from .core import annotate, annotate_spatial
from .models import AnnotationResult, CellTypeAnnotation, DeconvolutionResult, TopicAnnotation

__all__ = [
    "annotate",
    "annotate_spatial",
    "AnnotationResult",
    "CellTypeAnnotation",
    "DeconvolutionResult",
    "TopicAnnotation",
]
__version__ = "0.1.0"
