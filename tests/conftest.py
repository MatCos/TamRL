import pytest

from src.parser.tokenizer import TokenizerConfig


@pytest.fixture
def default_tokenizer_config() -> TokenizerConfig:
    return TokenizerConfig(
        max_length=512,
        cache_size=100,
        model_name="roberta-base",
        replace_var="none",
        key="name",
        id_range=50,
    )


@pytest.fixture
def no_cache_config() -> TokenizerConfig:
    return TokenizerConfig(
        max_length=512,
        cache_size=0,
        model_name="roberta-base",
        replace_var="none",
        key="name",
        id_range=50,
    )


@pytest.fixture
def replace_user_config() -> TokenizerConfig:
    return TokenizerConfig(
        max_length=512,
        cache_size=0,
        model_name="roberta-base",
        replace_var="user",
        key="name",
        id_range=50,
    )


@pytest.fixture
def replace_all_config() -> TokenizerConfig:
    return TokenizerConfig(
        max_length=512,
        cache_size=0,
        model_name="roberta-base",
        replace_var="all",
        key="name",
        id_range=50,
    )


def _leaf(name: str, type_: str = "node", **extra) -> dict:
    d = {"name": name, "type": type_, "args": []}
    d.update(extra)
    return d


def _node(name: str, args: list, type_: str = "fun", **extra) -> dict:
    d = {"name": name, "type": type_, "args": args}
    d.update(extra)
    return d


@pytest.fixture
def simple_pfm():
    return [("simplify", [])]


@pytest.fixture
def premise_pfm():
    tree = _node("KU", [_leaf("x", "msg")])
    return [("premise", [tree])]


@pytest.fixture
def multi_pfm():
    return [
        ("simplify", []),
        ("premise", [_node("KU", [_leaf("x", "msg")])]),
        ("chain", [_node("Chain", [_leaf("a", "node"), _leaf("b", "node")])]),
    ]


@pytest.fixture
def pfm_with_user_defined():
    tree = _node("KU", [_leaf("myVar", "msg", userDefined="True")])
    return [("premise", [tree])]


@pytest.fixture
def pfm_batch(simple_pfm, premise_pfm):
    return [simple_pfm, premise_pfm]


# Pre-processed (string args) fixtures for the fast encode path


@pytest.fixture
def simple_pfm_str():
    return [("simplify", ())]


@pytest.fixture
def premise_pfm_str():
    return [("premise", ("KU(x)",))]


@pytest.fixture
def multi_pfm_str():
    return [
        ("simplify", ()),
        ("premise", ("KU(x)",)),
        ("chain", ("Chain(a, b)",)),
    ]


@pytest.fixture
def pfm_batch_str(simple_pfm_str, premise_pfm_str):
    return [simple_pfm_str, premise_pfm_str]
