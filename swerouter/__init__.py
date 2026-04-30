"""SWERouterBench: dynamic SWE-bench Verified evaluation for per-step model routers.

Public surface (stable; anything not re-exported here is internal and may change):

- Router protocol and decision dataclasses: :mod:`swerouter.router`.
- Real per-model pricing and 4-bucket cost computation: :mod:`swerouter.pricing`.
- Wall-clock prompt cache simulation: :mod:`swerouter.cache`.
- Provider-side usage normalization: :mod:`swerouter.usage`.
- OpenAI-compatible chat client with cache_control injection: :mod:`swerouter.llm_client`.
- End-to-end run entrypoints and CLI: :mod:`swerouter.harness`, :mod:`swerouter.cli`.
- Leaderboard scoring (single metric: total USD spent): :mod:`swerouter.leaderboard`.

The benchmark itself (SWE-bench Verified) is consumed as a pip dependency
(`swebench>=2`) and never vendored inside this package.
"""

__version__ = "0.2.0"
