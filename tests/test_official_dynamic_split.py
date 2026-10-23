"""The dynamic track runs only the locked 100-case list."""

from __future__ import annotations

import pytest

from miniswerouter.harness.official_split import (
    assert_no_instances_outside_official,
    load_official_dynamic_instance_ids,
    select_official_dynamic_ids,
)
from miniswerouter.harness.run_eval import _resolve_instance_list


def test_official_list_is_the_locked_100():
    ids = load_official_dynamic_instance_ids()
    assert len(ids) == 100
    assert len(set(ids)) == 100
    assert sum(item.startswith("astropy__") for item in ids) == 18
    assert sum(item.startswith("django__") for item in ids) == 82
    assert ids == sorted(ids)


def test_default_selection_is_the_whole_official_list():
    official = load_official_dynamic_instance_ids()
    assert select_official_dynamic_ids(None, official=official) == official


def test_subset_keeps_official_order_and_drops_duplicates():
    official = load_official_dynamic_instance_ids()
    requested = [official[10], official[2], official[10]]
    assert select_official_dynamic_ids(requested, official=official) == [
        official[2],
        official[10],
    ]


def test_ids_outside_the_official_list_are_refused():
    official = load_official_dynamic_instance_ids()
    with pytest.raises(ValueError, match="outside that list"):
        select_official_dynamic_ids(
            ["django__django-11133", official[0]],
            official=official,
        )


def test_limit_only_truncates_the_official_list():
    official = load_official_dynamic_instance_ids()
    assert select_official_dynamic_ids(None, official=official, limit=3) == official[:3]
    with pytest.raises(ValueError, match="positive integer"):
        select_official_dynamic_ids(None, official=official, limit=0)


def test_score_guard_rejects_results_outside_the_list():
    official = load_official_dynamic_instance_ids()
    with pytest.raises(ValueError, match="remove 1 other result"):
        assert_no_instances_outside_official(
            [official[0], "matplotlib__matplotlib-13989"],
            official=official,
        )
    assert_no_instances_outside_official([official[0]], official=official)


def test_run_resolver_rejects_other_ids_without_loading_a_run():
    with pytest.raises(ValueError, match="outside that list"):
        _resolve_instance_list("princeton-nlp/SWE-bench_Verified", "test", ["django__django-11133"], None)
