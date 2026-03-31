# from .cal_metric import call_TEDS, call_BLEU, call_METEOR, call_Edit_dist, call_CDM, call_Move_dist
from registry.registry import METRIC_REGISTRY

from .cal_metric import (
    CallBLEU,
    CallCDM,
    CallCDMPlain,
    CallEditDist,
    CallMETEOR,
    CallTEDS,
)

__all__ = [
    "CallTEDS",
    "CallBLEU",
    "CallMETEOR",
    "CallEditDist",
    "CallCDM",
    "CallCDMPlain",
]

print("METRIC_REGISTRY: ", METRIC_REGISTRY.list_items())
