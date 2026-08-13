#!/usr/bin/env python3
"""Controller specialization for the v20b finite equality census."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import run_laion1m_v19_same_resident_paired_formal as core


core.MANIFEST_SCHEMA = (
    "laion1m-v20b-exact-equality-same-resident-paired-formal-manifest/v1"
)
core.SYSTEM_FREEZE_SCHEMA = "laion1m-exact-equality-v20b-system-freeze/v1"
core.POOL_SCHEMA = "laion1m-exact-equality-v20b-pool/v1"
core.SCIENCE_SCHEMA = "laion1m-exact-equality-v20b-science/v1"
core.QUERY_SCHEMA = "laion1m-v20b-exact-equality-formal-query-audit/v1"
core.SUMMARY_SCHEMA = "laion1m-v20b-exact-equality-formal-block-summary/v1"
core.CONTROLLER_SCHEMA = "laion1m_v20b_exact_equality_formal_controller_v1"

core.BACKENDS = dict(core.BACKENDS)
core.BACKENDS[("H", "equality")] = "complete_support_exact_topk"
core.H_EXACT_FAMILIES = ("equality", "range", "dnf2")
core.REQUIRE_EXACT_FAMILY_BINDING = True
core.PAIR1_MIN_H_OVER_S = 1.80
core.FIRST4_PAIR_GM_MIN_H_OVER_S = 1.90
core.FINAL8_PAIR_GM_MIN_H_OVER_S = 2.00
core.FINAL8_ONE_SIDED_P_MAX = 0.05
core.FINAL8_TWO_SIDED_P_MAX = 0.05
core.REQUIRED_SYSTEM_POPULATION_CONTRACT = {
    "exclude_v18_v19_timed_and_warmup_query_ids_all_families": True,
    "equality_formula_domain_cardinality": 200,
    "equality_timed_policy": "exactly_each_atom_once",
    "equality_formula_reuse_due_to_finite_domain": True,
    "complex_family_formula_disjoint_v18_v19": [
        "conjunction", "range", "dnf2"
    ],
    "selection_reads_structured_metadata_only": True,
    "query_vectors_read": False,
    "ground_truth_read_or_computed": False,
    "ann_results_or_performance_outcomes_read": False,
    "cross_preflight_required_before_candidate_enumeration": True,
}
core.REQUIRED_POOL_POPULATION_CONTRACT = {
    "equality_timed_population": "all_200_atoms_exactly_once",
    "equality_formula_disjointness": "not_applicable_finite_domain_census",
    "non_equality_formula_disjoint_families": [
        "conjunction", "range", "dnf2"
    ],
    "global_unique_query_base_ids": 864,
}
core.REQUIRED_SCIENCE_POPULATION_CONTRACT = {
    **core.REQUIRED_POOL_POPULATION_CONTRACT,
    "equality_warmup_role": (
        "non_estimand_fresh_identities_formula_reuse_allowed"
    ),
}
core.REQUIRED_EQUALITY_ATOM_COUNT = 200

_base_validate_activation_documents = core.validate_activation_documents


def _read_pinned_rows(
    pin: Any, expected_path: str, label: str
) -> list[dict[str, Any]]:
    if not isinstance(pin, dict) or set(pin) != {"path", "bytes", "mtime_ns"}:
        core.fail(f"v20b {label} stat-pin fields differ")
    if (
        type(pin["bytes"]) is not int
        or type(pin["mtime_ns"]) is not int
        or not 0 < pin["bytes"] <= 64 * 1024 * 1024
        or pin["mtime_ns"] <= 0
    ):
        core.fail(f"v20b {label} stat-pin values differ")
    path = Path(str(pin["path"])).resolve(strict=True)
    expected = Path(expected_path).resolve(strict=True)
    before = path.stat()
    if (
        path != expected
        or not path.is_file()
        or before.st_size != pin["bytes"]
        or before.st_mtime_ns != pin["mtime_ns"]
    ):
        core.fail(f"v20b {label} ordinary stat pin differs")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                core.fail(f"v20b {label}:{number} is not an object")
            rows.append(row)
    after = path.stat()
    if after.st_size != before.st_size or after.st_mtime_ns != before.st_mtime_ns:
        core.fail(f"v20b {label} changed while being read")
    return rows


def _validate_population_rows(
    timed: list[dict[str, Any]], warmup: list[dict[str, Any]]
) -> None:
    if len(timed) != 800 or len(warmup) != 64:
        core.fail("v20b timed/warmup request census differs")
    if Counter(str(row.get("family")) for row in timed) != Counter(
        {family: 200 for family in core.FAMILIES}
    ) or Counter(str(row.get("family")) for row in warmup) != Counter(
        {family: 16 for family in core.FAMILIES}
    ):
        core.fail("v20b family request census differs")
    equality_tokens = [
        row.get("primary_token_id")
        for row in timed if row.get("family") == "equality"
    ]
    if any(type(token) is not int for token in equality_tokens):
        core.fail("v20b timed equality atom type differs")
    equality_atoms = Counter(equality_tokens)
    if equality_atoms != Counter({token: 1 for token in range(200)}):
        core.fail("v20b timed equality atom census is not exactly once")
    all_rows = [*timed, *warmup]
    identities = [row.get("query_base_id") for row in all_rows]
    qids = [row.get("predicate_qid") for row in all_rows]
    if (
        any(type(value) is not int for value in identities)
        or any(type(value) is not int for value in qids)
        or len(set(identities)) != 864
        or len(set(qids)) != 864
    ):
        core.fail("v20b global query identity uniqueness differs")
    for role, rows in (("timed", timed), ("warmup", warmup)):
        for row in rows:
            family = str(row.get("family"))
            anchor = row.get("query_base_id")
            qid = row.get("predicate_qid")
            reference = row.get("query_vector_ref")
            expected = (
                "not_applicable_finite_domain_census"
                if family == "equality"
                else "disjoint_from_v18_v19_timed_warmup"
            )
            if (
                row.get("request_role") != role
                or type(anchor) is not int
                or type(qid) is not int
                or qid != 5_000_000 + anchor
                or not isinstance(reference, dict)
                or reference.get("container_role") != "base_vectors"
                or reference.get("row_index") != anchor
                or reference.get("base_id") != anchor
                or row.get("formula_disjointness") != expected
            ):
                core.fail("v20b row identity/role/formula binding differs")
            if family == "equality":
                token = row.get("primary_token_id")
                if (
                    type(token) is not int
                    or not 0 <= token < 200
                    or row.get("equality_atom_id") != token
                    or row.get("predicate") != f"A|{token}"
                    or row.get("equality_census_role")
                    != (
                        "timed_exactly_once"
                        if role == "timed" else "warmup_non_estimand"
                    )
                ):
                    core.fail("v20b equality row census binding differs")
            elif (
                row.get("equality_atom_id") is not None
                or row.get("equality_census_role") is not None
            ):
                core.fail("v20b complex row carries an equality census role")
    for family in ("conjunction", "range", "dnf2"):
        formulas = [
            row.get("predicate") for row in all_rows
            if row.get("family") == family
        ]
        if (
            any(not isinstance(value, str) for value in formulas)
            or len(formulas) != len(set(formulas))
        ):
            core.fail(f"v20b {family} formula multiplicity differs")


def _validate_population_bindings(
    system: dict[str, Any], pool: dict[str, Any], science: dict[str, Any]
) -> None:
    if system.get("fresh_population_contract") != (
        core.REQUIRED_SYSTEM_POPULATION_CONTRACT
    ):
        core.fail("v20b system population contract differs")
    if any(
        pool.get(key) != value
        for key, value in core.REQUIRED_POOL_POPULATION_CONTRACT.items()
    ) or any(
        science.get(key) != value
        for key, value in core.REQUIRED_SCIENCE_POPULATION_CONTRACT.items()
    ):
        core.fail("v20b finite equality population binding differs")


def validate_v20b_activation_documents(
    *, system: dict[str, Any], pool: dict[str, Any],
    science: dict[str, Any], **arguments: Any
) -> None:
    _base_validate_activation_documents(
        system=system, pool=pool, science=science, **arguments
    )
    _validate_population_bindings(system, pool, science)
    runner_arguments = arguments.get("runner_arguments")
    if not isinstance(runner_arguments, dict):
        core.fail("v20b runner arguments absent from population validation")
    timed = _read_pinned_rows(
        pool.get("timed_requests_stat_pin"),
        str(runner_arguments["queries"]),
        "timed requests",
    )
    warmup = _read_pinned_rows(
        pool.get("warmup_requests_stat_pin"),
        str(pool.get("warmup_requests_path")),
        "warmup requests",
    )
    _validate_population_rows(timed, warmup)


core.validate_activation_documents = validate_v20b_activation_documents


def query_free_self_test() -> int:
    current = core.validate_activation_documents
    try:
        core.validate_activation_documents = _base_validate_activation_documents
        result = core.query_free_self_test()
    finally:
        core.validate_activation_documents = current
    timed: list[dict[str, Any]] = []
    warmup: list[dict[str, Any]] = []
    next_identity = 0
    for role, target, per_family in (
        ("timed", timed, 200), ("warmup", warmup, 16)
    ):
        for family in core.FAMILIES:
            for index in range(per_family):
                target.append({
                    "request_role": role,
                    "family": family,
                    "primary_token_id": index if family == "equality" else 0,
                    "query_base_id": next_identity,
                    "predicate_qid": 5_000_000 + next_identity,
                    "predicate": f"{family}|{role}|{index}",
                    "query_vector_ref": {
                        "container_role": "base_vectors",
                        "row_index": next_identity,
                        "base_id": next_identity,
                    },
                    "formula_disjointness": (
                        "not_applicable_finite_domain_census"
                        if family == "equality"
                        else "disjoint_from_v18_v19_timed_warmup"
                    ),
                    "equality_atom_id": (
                        index if family == "equality" else None
                    ),
                    "equality_census_role": (
                        "timed_exactly_once"
                        if family == "equality" and role == "timed"
                        else "warmup_non_estimand"
                        if family == "equality" else None
                    ),
                })
                if family == "equality":
                    target[-1]["predicate"] = f"A|{index}"
                next_identity += 1
    _validate_population_rows(timed, warmup)
    duplicate_atom = [dict(row) for row in timed]
    equality_positions = [
        index for index, row in enumerate(duplicate_atom)
        if row["family"] == "equality"
    ]
    duplicate_atom[equality_positions[-1]]["primary_token_id"] = 198
    try:
        _validate_population_rows(duplicate_atom, warmup)
    except RuntimeError as error:
        if "atom census" not in str(error):
            raise
    else:
        core.fail("v20b duplicate equality atom counterexample was accepted")
    duplicate_identity = [dict(row) for row in warmup]
    duplicate_identity[-1]["query_base_id"] = timed[0]["query_base_id"]
    duplicate_identity[-1]["predicate_qid"] = timed[0]["predicate_qid"]
    try:
        _validate_population_rows(timed, duplicate_identity)
    except RuntimeError as error:
        if "identity uniqueness" not in str(error):
            raise
    else:
        core.fail("v20b duplicate identity/QID counterexample was accepted")
    valid_system = {
        "fresh_population_contract": dict(
            core.REQUIRED_SYSTEM_POPULATION_CONTRACT
        )
    }
    valid_pool = dict(core.REQUIRED_POOL_POPULATION_CONTRACT)
    valid_science = dict(core.REQUIRED_SCIENCE_POPULATION_CONTRACT)
    _validate_population_bindings(valid_system, valid_pool, valid_science)
    missing_pool = dict(valid_pool)
    del missing_pool["equality_timed_population"]
    try:
        _validate_population_bindings(valid_system, missing_pool, valid_science)
    except RuntimeError as error:
        if "population binding" not in str(error):
            raise
    else:
        core.fail("v20b missing population binding was accepted")
    with tempfile.TemporaryDirectory(prefix="laion_v20b_pin_fixture_") as root:
        path = Path(root) / "requests.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        value = path.stat()
        bad_pin = {
            "path": str(path.resolve()),
            "bytes": int(value.st_size) + 1,
            "mtime_ns": int(value.st_mtime_ns),
        }
        try:
            _read_pinned_rows(bad_pin, str(path.resolve()), "synthetic requests")
        except RuntimeError as error:
            if "ordinary stat pin" not in str(error):
                raise
        else:
            core.fail("v20b stat-pin drift counterexample was accepted")
    print(
        "SELF_TEST schema=laion1m_v20b_population_binding_v1 status=PASS "
        "real_query_files_opened=0 synthetic_pin_fixture_files_opened=1 "
        "performance_processes_started=0 "
        "equality_timed_atoms=200 exactly_each_atom_once=true "
        "duplicate_atom_counterexample=STOP "
        "duplicate_identity_qid_counterexample=STOP "
        "missing_population_binding_counterexample=STOP "
        "stat_pin_drift_counterexample=STOP global_unique_query_base_ids=864"
    )
    return result


if __name__ == "__main__":
    try:
        if "--query-free-self-test" in sys.argv[1:]:
            if len(sys.argv) != 2:
                core.fail("v20b self-test accepts no other arguments")
            raise SystemExit(query_free_self_test())
        raise SystemExit(core.main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"BLOCKED {error}", file=sys.stderr)
        raise SystemExit(2)
