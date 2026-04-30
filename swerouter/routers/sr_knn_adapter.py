"""``SemanticRouterKNNRouter``: drive per-step routing with vllm-project/semantic-router's
pretrained KNN model-selection artefact.

The upstream semantic-router Go service is not required; this adapter reuses
only the Python training sibling under
``semantic-router/src/training/model_selection/ml_model_selection/`` (CPU-only
scikit-learn + sentence-transformers path). It mirrors the approach already
documented in ``doc/semantic-router-on-commonrouterbench.md.md``.

Pipeline per step:

1. Flatten ``RouterContext.messages`` into ``role:\ncontent`` text.
2. Encode with Qwen3-Embedding-0.6B (1024-d) and concatenate a 14-d category
   one-hot (fixed to ``other`` for smoke; matches upstream default).
3. KNN (k=5, cosine) votes over its four training labels
   (``llama-3.2-1b|3b``, ``mistral-7b``, ``codellama-7b``).
4. Map the label to an official pool ``model_id`` via
   ``data/sr_knn_to_pool.json``. Anything else is fail-fast.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from swerouter.router import RouterContext, RouterDecision


_KNN_MODEL_CLS = None


def _load_knn_module(models_py_path: Path):
    """Import the upstream ``models.py`` once by path.

    The semantic-router ML training tree has no ``pyproject.toml`` and is not
    on ``sys.path``; loading by file path keeps us tolerant to whatever
    checkout layout the user has. Raises on any failure.
    """

    global _KNN_MODEL_CLS
    if _KNN_MODEL_CLS is not None:
        return _KNN_MODEL_CLS
    if not models_py_path.is_file():
        raise FileNotFoundError(f"semantic-router models.py not found: {models_py_path}")
    spec = importlib.util.spec_from_file_location(
        "swerouter_sr_knn_models", models_py_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build spec for {models_py_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["swerouter_sr_knn_models"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "KNNModel"):
        raise ImportError(f"{models_py_path} has no KNNModel class")
    _KNN_MODEL_CLS = module.KNNModel
    return _KNN_MODEL_CLS


# Ordered VSR domain one-hot used by upstream (see training README).
VSR_CATEGORIES: tuple[str, ...] = (
    "biology",
    "business",
    "chemistry",
    "computer science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "other",
    "philosophy",
    "physics",
    "psychology",
)


def _flatten_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
                    continue
                parts.append(json.dumps(block, ensure_ascii=False))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _flatten_messages_to_prompt(messages: tuple[Mapping[str, Any], ...]) -> str:
    """Match CommonRouterBench's classifier prompt shape: ``role:\\ncontent``."""

    blocks: list[str] = []
    for msg in messages:
        role = str(msg.get("role") or "user")
        text = _flatten_message_content(msg.get("content"))
        blocks.append(f"{role}:\n{text}")
    return "\n\n".join(blocks)


def _category_one_hot(category: str) -> np.ndarray:
    if category not in VSR_CATEGORIES:
        raise ValueError(
            f"unknown VSR category {category!r}; expected one of {VSR_CATEGORIES}"
        )
    vec = np.zeros(len(VSR_CATEGORIES), dtype=np.float32)
    vec[VSR_CATEGORIES.index(category)] = 1.0
    return vec


@dataclass
class SemanticRouterKNNRouter:
    """Adapter from SWERouterBench's per-step protocol to semantic-router's pretrained KNN.

    Parameters
    ----------
    knn_json_path
        Path to ``knn_model.json`` (feature_dim 1038, four labels) downloaded
        from ``abdallah1008/semantic-router-ml-models``.
    mapping_path
        Path to ``data/dynamic/sr_knn_to_pool.json`` (under ``TwinRouterBench/``). Every label the
        KNN might return must appear as a key, and every value must be a
        member of the active model pool.
    embedding_model
        Sentence-Transformers model ID. Upstream training used
        ``Qwen/Qwen3-Embedding-0.6B`` (1024-d); keep this default unless the
        KNN was retrained with a different embedder.
    embedding_device
        ``"cpu"`` (default), ``"cuda"``, or ``"mps"``.
    sr_repo_root
        Repository root for the checked-out ``semantic-router`` repo. The
        adapter loads ``<repo>/src/training/model_selection/ml_model_selection/models.py``
        from here to reuse its ``KNNModel`` class verbatim.
    label
        Human-readable identifier written into per-step ``rationale``.
    category
        Domain one-hot assignment per step. Kept as a single constant for the
        smoke run; ``other`` is upstream's neutral default. Extend to a live
        classifier once the smoke chain is validated.
    """

    knn_json_path: Path
    mapping_path: Path
    sr_repo_root: Path
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_device: str = "cpu"
    label: str = "sr_knn"
    category: str = "other"

    def __post_init__(self) -> None:
        knn_path = Path(self.knn_json_path)
        if not knn_path.is_file():
            raise FileNotFoundError(f"knn_model.json not found: {knn_path}")
        map_path = Path(self.mapping_path)
        if not map_path.is_file():
            raise FileNotFoundError(f"sr_knn_to_pool mapping not found: {map_path}")
        repo_root = Path(self.sr_repo_root)
        models_py = (
            repo_root
            / "src"
            / "training"
            / "model_selection"
            / "ml_model_selection"
            / "models.py"
        )

        with map_path.open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
        raw_map = doc.get("map")
        if not isinstance(raw_map, dict) or not raw_map:
            raise ValueError(f"{map_path} missing non-empty .map object")
        label_to_model: dict[str, str] = {
            str(k): str(v) for k, v in raw_map.items() if isinstance(v, str) and v
        }
        if not label_to_model:
            raise ValueError(f"{map_path}: .map has no string values")

        KNNModel = _load_knn_module(models_py)
        knn = KNNModel.load(str(knn_path))
        knn_labels = sorted(set(knn.model_names))
        missing = [lab for lab in knn_labels if lab not in label_to_model]
        if missing:
            raise ValueError(
                f"SemanticRouterKNNRouter[{self.label!r}]: knn_model.json has labels {missing} "
                f"that are not covered by mapping {map_path}"
            )

        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(self.embedding_model, device=self.embedding_device)
        expected_dim = 1024 + len(VSR_CATEGORIES)
        feature_dim = getattr(knn, "feature_dim", 0) or (
            len(knn.samples[0].feature_vector) if knn.samples else 0
        )
        if feature_dim != expected_dim:
            raise ValueError(
                f"SemanticRouterKNNRouter[{self.label!r}]: KNN feature_dim={feature_dim} "
                f"does not match expected {expected_dim} "
                f"(1024 Qwen3 + {len(VSR_CATEGORIES)} category one-hot)"
            )

        object.__setattr__(self, "_knn", knn)
        object.__setattr__(self, "_encoder", encoder)
        object.__setattr__(self, "_label_to_model", label_to_model)

    def _build_feature(self, messages: tuple[Mapping[str, Any], ...]) -> np.ndarray:
        prompt = _flatten_messages_to_prompt(messages)
        embedding = self._encoder.encode(
            [prompt],
            normalize_embeddings=False,
            convert_to_numpy=True,
        )[0].astype(np.float32)
        if embedding.shape[0] != 1024:
            raise ValueError(
                f"SemanticRouterKNNRouter[{self.label!r}]: embedding dim="
                f"{embedding.shape[0]}, expected 1024 from Qwen3-Embedding-0.6B"
            )
        one_hot = _category_one_hot(self.category)
        return np.concatenate([embedding, one_hot])

    def select(self, ctx: RouterContext) -> RouterDecision:
        started = time.perf_counter()
        feature = self._build_feature(ctx.messages)
        predicted_label = self._knn.predict(feature)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        mapping: dict[str, str] = getattr(self, "_label_to_model")
        model_id = mapping.get(predicted_label)
        if not model_id:
            raise ValueError(
                f"SemanticRouterKNNRouter[{self.label!r}] step={ctx.step_index}: "
                f"KNN returned unknown label {predicted_label!r}; mapping covers "
                f"{sorted(mapping.keys())}"
            )
        if model_id not in ctx.available_models:
            raise ValueError(
                f"SemanticRouterKNNRouter[{self.label!r}] step={ctx.step_index}: "
                f"mapped label={predicted_label!r} -> {model_id!r}, "
                f"not in pool {list(ctx.available_models)}"
            )

        rationale = (
            f"sr_knn label={predicted_label} category={self.category} "
            f"elapsed_ms={elapsed_ms:.1f}"
        )
        return RouterDecision(model_id=model_id, rationale=rationale)

    @classmethod
    def from_cli_args(
        cls,
        *,
        knn_json_path: str,
        mapping_path: str,
        sr_repo_root: str,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        embedding_device: str = "cpu",
        label: str = "sr_knn",
        category: str = "other",
    ) -> "SemanticRouterKNNRouter":
        """CLI-friendly factory; all kwargs are forwarded as strings."""

        return cls(
            knn_json_path=Path(knn_json_path),
            mapping_path=Path(mapping_path),
            sr_repo_root=Path(sr_repo_root),
            embedding_model=embedding_model,
            embedding_device=embedding_device,
            label=label,
            category=category,
        )
