"""MiniSWERouterBench: SWERouterBench's dynamic router evaluation protocol
rewired to run on top of mini-swe-agent's scaffold.

Public surface (stable):

- :mod:`miniswerouter.harness.env` — ``SwebenchContainerEnv`` (mini Environment).
- :mod:`miniswerouter.agent.model` — ``RouterAwareModel`` (mini Model with per-step router dispatch + SWERouterBench pricing).
- :mod:`miniswerouter.agent.agent` — ``MiniRouterAgent`` (subclass of ``minisweagent.agents.default.DefaultAgent``).
- :mod:`miniswerouter.harness.run_instance` — end-to-end per-instance runner.
- :mod:`miniswerouter.harness.run_eval` — concurrent multi-instance runner.
- :mod:`miniswerouter.cli` — ``miniswerouterbench`` command-line entrypoint.

Scoring/rendering is delegated to :mod:`swerouter.leaderboard` so the two
scaffolds produce byte-compatible leaderboard files.
"""

__version__ = "0.1.0"
