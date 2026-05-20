import os
from dataclasses import dataclass
from enum import Enum

import pytest

from src.utils.cache import LRUCache
from src.utils.load import (
    LemmaType,
    TheoryConfig,
    extract_lemma_type,
    load_spthy_file,
    open_spthy_file,
    parse_diff_arg,
    parse_lemmas,
    parse_theory_name,
    remove_c_style_comments,
)
from src.utils.utils import (
    dataclass_to_dict,
    eval_num,
    get_slurm_id,
    make_hashable,
)


# ---------------------------------------------------------------------------
# LRUCache
# ---------------------------------------------------------------------------


class TestLRUCache:
    def test_invalid_capacity(self):
        for bad in (0, -1, 3.5):
            with pytest.raises(ValueError):
                LRUCache(bad)

    def test_basic_ops(self):
        cache = LRUCache(3)
        cache["a"] = 1
        assert cache["a"] == 1
        assert cache.get("a") == 1
        assert cache.get("missing") is None
        assert cache.get("missing", 42) == 42
        with pytest.raises(KeyError):
            _ = cache["missing"]

    def test_eviction_respects_access_order(self):
        cache = LRUCache(2)
        cache["a"] = 1
        cache["b"] = 2
        _ = cache["a"]  # refresh "a"
        cache["c"] = 3  # should evict "b", not "a"
        assert "a" in cache
        assert "b" not in cache
        assert cache["c"] == 3

    def test_overwrite_refreshes_order(self):
        cache = LRUCache(2)
        cache["a"] = 1
        cache["b"] = 2
        cache["a"] = 10
        cache["c"] = 3  # should evict "b"
        assert cache["a"] == 10
        assert "b" not in cache

    def test_hit_miss_tracking(self):
        cache = LRUCache(4)
        assert cache.hit_rate == 0.0
        assert cache.usage == 0.0

        cache["a"] = 1
        cache["b"] = 2
        assert cache.usage == 0.5

        cache.check("a")  # hit
        cache.check("missing")  # miss
        assert cache.hits == 1
        assert cache.misses == 1
        assert cache.hit_rate == 0.5


# ---------------------------------------------------------------------------
# load.py
# ---------------------------------------------------------------------------


SAMPLE_THEORY = """\
theory Test
begin

lemma myLemma:
  all-traces
  "some formula"

lemma inlineExists: exists-trace
  "some formula"

lemma existsLemma:
  exists-trace
  "some formula"

lemma implicitForall:
  "some formula without quantifier"

end
"""


class TestRemoveComments:
    def test_removes_block_and_line_comments(self):
        assert remove_c_style_comments("a /* b */ c") == "a  c"
        assert remove_c_style_comments("a /* \n b \n */ c") == "a  c"
        assert remove_c_style_comments("a // b\nc") == "a \nc"
        assert remove_c_style_comments("hello world") == "hello world"


class TestSpthyParsing:
    def test_lemma_type_str(self):
        assert str(LemmaType.EXISTS) == "exists-trace"
        assert str(LemmaType.FORALL) == "all-traces"

    def test_theory_config_frozen(self):
        cfg = TheoryConfig("path", "name", False, [])
        with pytest.raises(AttributeError):
            cfg.theory_path = "other"

    def test_parse_theory_name(self):
        assert parse_theory_name("theory MyTheory\nbegin\n", "t.spthy") == "MyTheory"
        with pytest.raises(ValueError):
            parse_theory_name("no theory here", "t.spthy")

    def test_parse_diff_arg(self):
        assert parse_diff_arg("rule: diff(a, b)") is True
        assert parse_diff_arg("rule: a + b") is False

    def test_extract_lemma_types(self):
        assert extract_lemma_type(SAMPLE_THEORY, "myLemma", "t.spthy") == LemmaType.FORALL
        assert extract_lemma_type(SAMPLE_THEORY, "existsLemma", "t.spthy") == LemmaType.EXISTS
        assert extract_lemma_type(SAMPLE_THEORY, "implicitForall", "t.spthy") == LemmaType.FORALL
        assert extract_lemma_type(SAMPLE_THEORY, "inlineExists", "t.spthy") == LemmaType.EXISTS
        with pytest.raises(ValueError):
            extract_lemma_type(SAMPLE_THEORY, "noSuchLemma", "t.spthy")

    def test_parse_lemmas(self):
        names = [name for name, _ in parse_lemmas(SAMPLE_THEORY, "t.spthy")]
        assert set(names) == {"myLemma", "inlineExists", "existsLemma", "implicitForall"}


class TestSpthyFileIO:
    def test_open_and_filter(self, tmp_path):
        (tmp_path / "a.spthy").write_text("theory A /* comment */\nbegin\nend")
        (tmp_path / "b.spthy").write_text("theory B\nbegin\nend")
        content, _ = open_spthy_file(str(tmp_path), "a.spthy")
        assert "theory A" in content
        assert "/* comment */" not in content  # comments stripped
        content_b, _ = open_spthy_file(str(tmp_path), "b.spthy")
        assert "theory B" in content_b

    def test_open_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            open_spthy_file(str(tmp_path))

    def test_load_spthy_file(self, tmp_path):
        (tmp_path / "proto.spthy").write_text(
            "theory MyProto\nbegin\n"
            "rule: diff(a, b)\n"
            "lemma sec:\n  all-traces\n  \"f\"\n"
            "end\n"
        )
        name, path, diff, lemmas = load_spthy_file(str(tmp_path))
        assert name == "MyProto"
        assert diff is True
        assert any(n == "sec" for n, _ in lemmas)


# ---------------------------------------------------------------------------
# utils.py
# ---------------------------------------------------------------------------


class TestMakeHashable:
    def test_converts_containers_to_tuples(self):
        assert make_hashable(42) == 42
        assert make_hashable("abc") == "abc"
        assert make_hashable([1, 2, 3]) == (1, 2, 3)
        assert make_hashable({"b": 2, "a": 1}) == (("a", 1), ("b", 2))
        assert make_hashable({3, 1, 2}) == (1, 2, 3)
        # nested structures should be fully hashable
        hash(make_hashable({"a": [1, {"b": 2}]}))

    def test_unhashable_fallback(self):
        class Unhashable:
            __hash__ = None
        assert isinstance(make_hashable(Unhashable()), str)


class TestDataclassToDict:
    def test_flat_nested_and_special_types(self):
        class Color(Enum):
            RED = "red"

        @dataclass(frozen=True)
        class Inner:
            val: int

        @dataclass(frozen=True)
        class Outer:
            inner: Inner
            name: str
            items: list
            color: Color

        result = dataclass_to_dict(Outer(Inner(5), "test", [10, 20], Color.RED))
        assert result == {
            "inner/val": 5,
            "name": "test",
            "items/0": 10,
            "items/1": 20,
            "color": "Color.RED",
        }

    def test_custom_separator(self):
        @dataclass(frozen=True)
        class Inner:
            val: int

        @dataclass(frozen=True)
        class Outer:
            inner: Inner

        assert dataclass_to_dict(Outer(Inner(5)), separator=".") == {"inner.val": 5}

    def test_list_of_dataclasses(self):
        @dataclass(frozen=True)
        class Item:
            val: int

        @dataclass(frozen=True)
        class Container:
            items: list

        result = dataclass_to_dict(Container([Item(1), Item(2)]))
        assert result == {"items/0/val": 1, "items/1/val": 2}


class TestMiscUtils:
    def test_eval_num(self, tmp_path):
        (tmp_path / "eval_1").mkdir()
        (tmp_path / "eval_2").mkdir()
        (tmp_path / "other").mkdir()
        assert eval_num(str(tmp_path)) == 2
        assert eval_num(str(tmp_path / "other")) == 0

    def test_get_slurm_id(self, monkeypatch):
        monkeypatch.delenv("SLURM_ARRAY_JOB_ID", raising=False)
        monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        assert get_slurm_id() is None

        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        assert get_slurm_id() == "12345"

        monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "100")
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "5")
        assert get_slurm_id() == "100_5"
