"""formulate -- active learning for polymer formulation.

Find the best copolymer formulation in a design space using as few experiments as
possible: a surrogate model plus an acquisition function proposes the next
formulation to synthesise, an oracle scores it, and the loop repeats under a hard
experimental budget. The headline is the money plot -- best property found vs.
number of experiments, active learning against random screening.

The oracle is the controlled copolymer-Tg model from ``copolybench``; the
surrogate uncertainty and gradient-boosting baseline come from ``polytools``. This
is Project 3, and it composes the earlier two into the loop an industrial ML
chemist actually runs.
"""

from .acquisition import ACQUISITIONS, acquisition_scores
from .design_space import DesignSpace, build_design_space
from .experiment import ExperimentSummary, experiments_to_target, run_experiment
from .loop import CampaignResult, run_campaign
from .surrogate import EnsembleSurrogate, GPSurrogate

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DesignSpace", "build_design_space",
    "GPSurrogate", "EnsembleSurrogate",
    "ACQUISITIONS", "acquisition_scores",
    "CampaignResult", "run_campaign",
    "ExperimentSummary", "run_experiment", "experiments_to_target",
]
