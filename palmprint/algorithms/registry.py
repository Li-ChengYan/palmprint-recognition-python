"""Algorithm registry for built-in palmprint methods."""

from __future__ import annotations

from palmprint.algorithms.bocv import BOCVAlgorithm
from palmprint.algorithms.cc import CCAlgorithm
from palmprint.algorithms.doc import DOCAlgorithm
from palmprint.algorithms.don import DoNAlgorithm
from palmprint.algorithms.drcc import DRCCAlgorithm
from palmprint.algorithms.ebocv import EBOCVAlgorithm
from palmprint.algorithms.edm import EDMAlgorithm
from palmprint.algorithms.fastcc import FastCCAlgorithm
from palmprint.algorithms.fc import FCAlgorithm
from palmprint.algorithms.mtcc import MTCCAlgorithm
from palmprint.algorithms.oc import OCAlgorithm
from palmprint.algorithms.pc import PCAlgorithm
from palmprint.algorithms.rloc import FastRLOCAlgorithm, RLOCAlgorithm

_ALGORITHMS = {
    "PC": PCAlgorithm,
    "FC": FCAlgorithm,
    "CC": CCAlgorithm,
    "FastCC": FastCCAlgorithm,
    "OC": OCAlgorithm,
    "RLOC": RLOCAlgorithm,
    "FastRLOC": FastRLOCAlgorithm,
    "BOCV": BOCVAlgorithm,
    "EBOCV": EBOCVAlgorithm,
    "DOC": DOCAlgorithm,
    "DRCC": DRCCAlgorithm,
    "EDM": EDMAlgorithm,
    "MTCC": MTCCAlgorithm,
    "DoN": DoNAlgorithm,
}


def list_algorithms() -> list[str]:
    """Return all supported algorithm names in registry order."""

    return list(_ALGORITHMS.keys())


def get_algorithm(name: str):
    """Instantiate one algorithm by name."""

    if name not in _ALGORITHMS:
        available = ", ".join(list_algorithms())
        raise ValueError(f"Unknown algorithm {name!r}. Available algorithms: {available}.")
    return _ALGORITHMS[name]()
