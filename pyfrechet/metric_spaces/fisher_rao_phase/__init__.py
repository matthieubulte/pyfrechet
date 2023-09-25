try:
    import skfda, fdasrsf
    from .fisher_rao_phase import FisherRaoPhase
    has_fda = True
except ImportError:
    has_fda = False