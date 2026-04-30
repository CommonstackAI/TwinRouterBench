"""Reference routers shipped with SWERouterBench.

Each router implements the :class:`swerouter.router.Router` protocol and is
safe to pass directly to :mod:`swerouter.harness.run_eval`.
"""

from swerouter.routers.always_model import AlwaysModelRouter
from swerouter.routers.gold_tier import GoldTierRouter
from swerouter.routers.round_robin import RoundRobinRouter
from swerouter.routers.tier_from_crb import TierFromCRBRouter

__all__ = [
    "AlwaysModelRouter",
    "GoldTierRouter",
    "RoundRobinRouter",
    "TierFromCRBRouter",
]
