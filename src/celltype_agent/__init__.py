"""celltype-agent: automated cell type annotation for single-cell & spatial genomics."""

from .core import annotate
from .models import AnnotationResult, CellTypeAnnotation

__all__ = ["annotate", "AnnotationResult", "CellTypeAnnotation"]
__version__ = "0.1.0"
