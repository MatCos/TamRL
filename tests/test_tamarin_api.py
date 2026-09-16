"""Integration tests that talk to a real Tamarin server over the HTTP API.

Every other test module mocks the prover away, so nothing else in the suite
covers `src/client/client.py` or the API of our Tamarin fork. These tests start
an actual `tamarin-prover-json interactive` server on the Tutorial theory,
prove all four of its lemmas by always applying the proof method Tamarin ranks
first, and let the server check the resulting proofs.

The module is skipped when `tamarin-prover-json` is not on `PATH`, and takes
about two seconds when it is.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from src.client.client import TamarinClient
from src.utils.utils import (
    CASE_NAME,
    COMPLETE_PROOF,
    CONTRADICTORY,
    ENCODED_METHODS,
    ENCODED_SYS,
    PROOF_METHODS,
    PROOF_METHOD_NAME,
    SOLVED,
    STATUS,
    THEORY_INDEX,
    THEORY_KIND,
    TRACE_FOUND,
)

THEORY_PATH = (
    Path(__file__).resolve().parents[1]
    / "eval"
    / "Tutorial"
    / "original"
    / "Tutorial.spthy"
)
THEORY_NAME = "Tutorial"

# Lemma, expected proof status, and expected proof size when always applying
# the first-ranked proof method. The sizes are those of Tamarin 1.11.0, the
# version shipped in the artifact image.
TUTORIAL_LEMMAS = [
    ("Client_session_key_secrecy", COMPLETE_PROOF, 5),
    ("Client_auth", COMPLETE_PROOF, 11),
    ("Client_auth_injective", COMPLETE_PROOF, 15),
    # exists-trace lemma: a trace is what we are looking for here
    ("Client_session_key_honest_setup", TRACE_FOUND, 11),
]

# The greedy proofs above need at most 11 steps; the cap only stops runaway
# searches from hanging the test suite.
MAX_STEPS = 100

MISSING_PROVER = (
    "tamarin-prover-json was not found on PATH, so the Tamarin API cannot be tested."
)


@pytest.fixture(scope="module")
def client():
    """One Tamarin server for the whole module; the API itself is stateless."""
    if shutil.which("tamarin-prover-json") is None:
        # test_tamarin_prover_is_installed already reports this as a failure.
        pytest.skip(MISSING_PROVER)
    tamarin = TamarinClient(
        str(THEORY_PATH),
        suppress_output=True,
        theory_name=THEORY_NAME,
        diff_arg=False,
    )
    yield tamarin
    tamarin.stop_process()


@pytest.fixture(scope="module")
def theory(client) -> tuple[str, int]:
    """The (kind, index) pair addressing the Tutorial theory on the server."""
    entry = client.server_overview[THEORY_NAME]
    return entry[THEORY_KIND], entry[THEORY_INDEX]


def _initial_system(client, theory, lemma: str) -> dict[str, Any]:
    kind, index = theory
    response = client.get_initial_constraint_system(kind, index, lemma)
    assert (
        response.ok
    ), f"{lemma}: initial constraint system failed: {response.text[:200]}"
    return json.loads(response.text)


def _prove_greedily(
    client, theory, lemma: str, system: dict[str, Any], budget: list[int]
) -> tuple[dict[str, Any], int]:
    """Applies the first-ranked proof method until every branch closes.

    Returns the proof tree in the format `check_proof` expects — the same one
    `MCTS.extract_proof_tree` produces — and its size in proof steps.
    """
    if system[STATUS] == SOLVED:
        return {"action": "solved", "children": {}}, 1
    if system[STATUS] == CONTRADICTORY:
        return {"action": "contradictory", "children": {}}, 1

    assert budget[0] > 0, f"{lemma}: not proved within {MAX_STEPS} proof steps"
    budget[0] -= 1

    kind, index = theory
    method = system[ENCODED_METHODS][0]
    response = client.apply_proofmethod_to_system(
        kind, index, lemma, method, system[ENCODED_SYS]
    )
    assert response.ok, f"{lemma}: applying proof method failed: {response.text[:200]}"

    children, size = {}, 1
    for case in json.loads(response.text):
        subtree, subsize = _prove_greedily(
            client, theory, lemma, case["system"], budget
        )
        children[case[CASE_NAME]] = subtree
        size += subsize
    return {"action": method, "children": children}, size


def test_tamarin_prover_is_installed():
    """Reports the missing prover as a plain failure, before any fixture runs."""
    assert shutil.which("tamarin-prover-json") is not None, MISSING_PROVER


def test_server_overview_lists_the_tutorial_theory(client):
    overview = client.server_overview
    assert THEORY_NAME in overview, f"loaded theories: {list(overview)}"
    assert overview[THEORY_NAME][THEORY_KIND] == "trace"


@pytest.mark.parametrize("lemma", [lemma for lemma, _, _ in TUTORIAL_LEMMAS])
def test_initial_constraint_system_offers_proof_methods(client, theory, lemma):
    system = _initial_system(client, theory, lemma)

    assert system[STATUS] not in (SOLVED, CONTRADICTORY)
    assert system[ENCODED_SYS]
    assert len(system[PROOF_METHODS]) == len(system[ENCODED_METHODS])
    assert system[PROOF_METHODS], "no proof method offered for the initial system"
    assert system[PROOF_METHODS][0][PROOF_METHOD_NAME] == "Simplify"


@pytest.mark.parametrize("lemma,expected_status,expected_size", TUTORIAL_LEMMAS)
def test_greedy_proof_is_accepted_by_tamarin(
    client, theory, lemma, expected_status, expected_size
):
    kind, index = theory
    tree, size = _prove_greedily(
        client, theory, lemma, _initial_system(client, theory, lemma), [MAX_STEPS]
    )
    assert size == expected_size

    response = client.check_proof(kind, index, lemma, json.dumps(tree))
    assert response.ok, f"{lemma}: check_proof failed: {response.text[:200]}"
    result = json.loads(response.text)

    assert result["proofStatus"] == expected_status
    assert result["proofSize"] == size
    assert f"lemma {lemma}" in result["proofFile"]
