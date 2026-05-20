import random

import pytest
import torch

from src.datasets.tokenizer import HFTokenizerWrapper
from src.parser.tokenizer import TokenizerConfig
from tests.conftest import _leaf, _node

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestInit:
    def test_creates_wrapper(self, default_tokenizer_config):
        wrapper = HFTokenizerWrapper(default_tokenizer_config)
        assert wrapper.tokenizer is not None
        assert wrapper.tokenizer.pad_token is not None
        assert wrapper.str_key == "name"
        assert wrapper.str_indent is False
        assert wrapper.max_length == 512
        assert wrapper.cache is not None
        assert wrapper.cache.capacity == 100

    def test_cache_disabled(self, no_cache_config):
        wrapper = HFTokenizerWrapper(no_cache_config)
        assert wrapper.cache is None


# ---------------------------------------------------------------------------
# _to_batch
# ---------------------------------------------------------------------------


class TestToBatch:
    def test_normalizes_input(self, default_tokenizer_config, simple_pfm, pfm_batch):
        wrapper = HFTokenizerWrapper(default_tokenizer_config)
        assert len(wrapper._to_batch(simple_pfm)) == 1
        assert wrapper._to_batch(pfm_batch) is pfm_batch


# ---------------------------------------------------------------------------
# _parse_sample
# ---------------------------------------------------------------------------


class TestParseSample:
    def test_empty_args(self, default_tokenizer_config, simple_pfm):
        wrapper = HFTokenizerWrapper(default_tokenizer_config)
        trees, labels = wrapper._parse_sample(simple_pfm, None)
        assert len(trees) == 1
        assert trees[0][0] == "simplify"
        assert trees[0][1] == []
        assert labels == set()

    def test_parses_fol_tree(self, default_tokenizer_config, premise_pfm):
        wrapper = HFTokenizerWrapper(default_tokenizer_config)
        trees, labels = wrapper._parse_sample(premise_pfm, None)
        from src.dtypes.fol_tree import FOLTree

        assert isinstance(trees[0][1][0], FOLTree)

    def test_int_args_become_strings(self, default_tokenizer_config):
        sample = [("premise", [42])]
        wrapper = HFTokenizerWrapper(default_tokenizer_config)
        trees, _ = wrapper._parse_sample(sample, None)
        assert trees[0][1] == ["42"]

    def test_collects_labels(
        self,
        replace_user_config,
        replace_all_config,
        pfm_with_user_defined,
        premise_pfm,
    ):
        wrapper = HFTokenizerWrapper(replace_user_config)
        _, labels = wrapper._parse_sample(pfm_with_user_defined, None)
        names = {name for name, _ in labels}
        assert "myVar" in names

        wrapper = HFTokenizerWrapper(replace_all_config)
        _, labels = wrapper._parse_sample(premise_pfm, None)
        assert len(labels) > 0


# ---------------------------------------------------------------------------
# _build_rename_map
# ---------------------------------------------------------------------------


class TestBuildRenameMap:
    def test_empty_labels(self, default_tokenizer_config):
        wrapper = HFTokenizerWrapper(default_tokenizer_config)
        assert wrapper._build_rename_map(set()) == {}

    def test_rename_format(self, default_tokenizer_config):
        wrapper = HFTokenizerWrapper(default_tokenizer_config)
        labels = {("myVar", "msg"), ("otherVar", "node")}
        random.seed(42)
        rename = wrapper._build_rename_map(labels)
        assert len(rename) == 2
        for name, replacement in rename.items():
            parts = replacement.split(".")
            assert len(parts) == 2
            assert parts[1].isdigit()


# ---------------------------------------------------------------------------
# _trees_to_strings
# ---------------------------------------------------------------------------


class TestTreesToStrings:
    def test_string_passthrough(self, default_tokenizer_config):
        wrapper = HFTokenizerWrapper(default_tokenizer_config)
        trees = [("simplify", []), ("premise", ["42"])]
        result = wrapper._trees_to_strings(trees, None)
        assert result == [("simplify", ()), ("premise", ("42",))]

    def test_fol_tree_to_str(self, default_tokenizer_config, premise_pfm):
        wrapper = HFTokenizerWrapper(default_tokenizer_config)
        trees, _ = wrapper._parse_sample(premise_pfm, None)
        result = wrapper._trees_to_strings(trees, None)
        assert isinstance(result[0][1][0], str)
        assert len(result[0][1][0]) > 0


# ---------------------------------------------------------------------------
# get_proofmethods
# ---------------------------------------------------------------------------


class TestGetProofmethods:
    def test_batch_output(self, default_tokenizer_config, pfm_batch):
        wrapper = HFTokenizerWrapper(default_tokenizer_config)
        result = wrapper.get_proofmethods(pfm_batch)
        assert len(result) == len(pfm_batch)
        for sample_result, sample_input in zip(result, pfm_batch):
            assert len(sample_result) == len(sample_input)
            for method, objs in sample_result:
                assert isinstance(method, str)
                assert isinstance(objs, tuple)

    def test_simplify_has_no_args(self, default_tokenizer_config, simple_pfm):
        wrapper = HFTokenizerWrapper(default_tokenizer_config)
        result = wrapper.get_proofmethods([simple_pfm])
        assert result[0][0] == ("simplify", ())


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_returns_tokens_with_eos(self, default_tokenizer_config, simple_pfm):
        wrapper = HFTokenizerWrapper(default_tokenizer_config)
        result = wrapper.tokenize([simple_pfm])
        tokens = result[0][0]
        assert all(isinstance(t, str) for t in tokens)
        eos = wrapper.tokenizer.eos_token or "[EOS]"
        assert tokens[-1] == eos

    def test_multi_pfm(self, default_tokenizer_config, multi_pfm):
        wrapper = HFTokenizerWrapper(default_tokenizer_config)
        result = wrapper.tokenize([multi_pfm])
        assert len(result[0]) == 3

    def test_max_length_respected(self, premise_pfm):
        config = TokenizerConfig(
            max_length=10,
            cache_size=0,
            model_name="roberta-base",
            replace_var="none",
            key="name",
            id_range=50,
        )
        wrapper = HFTokenizerWrapper(config)
        tokens = wrapper.tokenize([premise_pfm])[0][0]
        assert len(tokens) <= 10


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------


class TestEncode:
    def test_consistent_with_tokenize(self, default_tokenizer_config, premise_pfm):
        wrapper = HFTokenizerWrapper(default_tokenizer_config)
        tokens = wrapper.tokenize([premise_pfm])[0][0]
        ids = wrapper.encode([premise_pfm])[0][0]
        assert all(isinstance(i, int) for i in ids)
        assert ids == wrapper.tokenizer.convert_tokens_to_ids(tokens)


# ---------------------------------------------------------------------------
# _encode_fast
# ---------------------------------------------------------------------------


class TestEncodeFast:
    def test_produces_int_ids(self, default_tokenizer_config, pfm_batch_str):
        wrapper = HFTokenizerWrapper(default_tokenizer_config)
        result = wrapper._encode_fast(pfm_batch_str)
        assert len(result) == len(pfm_batch_str)
        for sample_result, sample_input in zip(result, pfm_batch_str):
            assert len(sample_result) == len(sample_input)
            for ids in sample_result:
                assert all(isinstance(i, int) for i in ids)

    def test_cache_populated_and_hit(self, default_tokenizer_config, simple_pfm_str):
        wrapper = HFTokenizerWrapper(default_tokenizer_config)
        assert len(wrapper.cache) == 0
        wrapper._encode_fast([simple_pfm_str])
        assert len(wrapper.cache) > 0
        hits_before = wrapper.cache.hits
        wrapper._encode_fast([simple_pfm_str])
        assert wrapper.cache.hits > hits_before

    def test_no_cache_still_works(self, no_cache_config, premise_pfm_str):
        wrapper = HFTokenizerWrapper(no_cache_config)
        result = wrapper._encode_fast([premise_pfm_str])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# __call__
# ---------------------------------------------------------------------------


class TestCall:
    def test_returns_batch_and_slices(
        self, default_tokenizer_config, multi_pfm_str, premise_pfm_str
    ):
        wrapper = HFTokenizerWrapper(default_tokenizer_config)
        pfm_batch = [multi_pfm_str, premise_pfm_str]
        batch, slices = wrapper(pfm_batch, return_tensors=None)
        assert "input_ids" in batch
        total = sum(len(s) for s in pfm_batch)
        assert len(batch["input_ids"]) == total
        assert slices[0] == 0
        assert slices[-1] == total

    def test_pt_tensors(self, default_tokenizer_config, simple_pfm_str):
        wrapper = HFTokenizerWrapper(default_tokenizer_config)
        batch, slices = wrapper(simple_pfm_str, return_tensors="pt")
        assert isinstance(batch["input_ids"], torch.Tensor)
        assert isinstance(slices, torch.Tensor)

    def test_attention_mask(self, default_tokenizer_config, pfm_batch_str):
        wrapper = HFTokenizerWrapper(default_tokenizer_config)
        batch, _ = wrapper(
            pfm_batch_str, return_attention_mask=True, return_tensors="pt"
        )
        assert "attention_mask" in batch

    def test_replace_var_path(self, replace_user_config, pfm_with_user_defined):
        """When replace_var='user', uses the slow encode path."""
        wrapper = HFTokenizerWrapper(replace_user_config)
        batch, slices = wrapper([pfm_with_user_defined], return_tensors=None)
        assert "input_ids" in batch

    def test_max_length_none(self, simple_pfm_str):
        config = TokenizerConfig(
            max_length=None,
            cache_size=0,
            model_name="roberta-base",
            replace_var="none",
            key="name",
            id_range=50,
        )
        wrapper = HFTokenizerWrapper(config)
        batch, _ = wrapper(simple_pfm_str, return_tensors=None)
        assert "input_ids" in batch


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_batch(self, default_tokenizer_config):
        wrapper = HFTokenizerWrapper(default_tokenizer_config)
        assert wrapper.get_proofmethods([]) == []

    def test_multiple_trees_per_method(self, default_tokenizer_config):
        tree1 = _node("KU", [_leaf("x", "msg")])
        tree2 = _node("KU", [_leaf("y", "msg")])
        sample = [("premise", [tree1, tree2])]
        wrapper = HFTokenizerWrapper(default_tokenizer_config)
        result = wrapper.get_proofmethods([sample])
        assert len(result[0][0][1]) == 2

    def test_batch_determinism(self, default_tokenizer_config, pfm_batch):
        wrapper = HFTokenizerWrapper(default_tokenizer_config)
        assert wrapper.encode(pfm_batch) == wrapper.encode(pfm_batch)
