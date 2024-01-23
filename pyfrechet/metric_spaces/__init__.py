from .metric_space import MetricSpace
from .metric_data import MetricData
from .euclidean import Euclidean
from .sphere import Sphere
from .correlation import CorrFrobenius
from .wasserstein_1d import Wasserstein1D
from .network import NetworkCholesky
from .riemannian_manifold import RiemannianManifold
from .log_cholesky import LogCholesky

from .fisher_rao_phase import has_fda
if has_fda:
    from .fisher_rao_phase import FisherRaoPhase
    