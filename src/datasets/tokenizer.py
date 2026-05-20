import random
import sys

import torch
from transformers import AutoTokenizer

from src.dtypes.fol_tree import FOLTree
from src.parser import TokenizerConfig
from src.utils.cache import LRUCache
from src.utils.utils import make_hashable

sys.setrecursionlimit(10000)


class HFTokenizerWrapper:

    def __init__(
        self,
        tokenizer_config: TokenizerConfig,
        **kwargs,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_config.model_name, **kwargs
        )
        # standard practice to use the eos token if no pad token is defined
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.str_indent: bool = False
        self.str_key: str | None = tokenizer_config.key
        self.replace_var = tokenizer_config.replace_var
        self.id_range = tokenizer_config.id_range
        self.max_length = tokenizer_config.max_length
        self.cache: LRUCache[str, list[int]] | None = (
            LRUCache(capacity=tokenizer_config.cache_size)
            if tokenizer_config.cache_size > 0
            else None
        )

    def _parse_sample(
        self,
        sample: list[tuple[str, list[dict]]],
        info_sample: tuple[str, int] | None,
    ) -> tuple[list[tuple[str, list[FOLTree | str]]], set[tuple[str, str]]]:
        """Parse proof method dicts into FOLTrees, collecting labels for replacement."""
        trees: list[tuple[str, list[FOLTree | str]]] = []
        replace_labels: set[tuple[str, str]] = set()

        for method, method_objs in sample:
            parsed: list[FOLTree | str] = []
            for method_obj in method_objs:
                try:
                    if isinstance(method_obj, int):
                        parsed.append(str(method_obj))
                    else:
                        tree = FOLTree.from_expression(method_obj, {})
                        tree.clean_heristic_names()
                        if self.replace_var == "user":
                            replace_labels.update(tree.user_defined)
                        elif self.replace_var == "all":
                            replace_labels.update(tree.labels)
                        parsed.append(tree)
                except RecursionError:
                    print(
                        f"Warning: RecursionError when parsing method {method}. Skipping."
                    )
                    if info_sample is not None:
                        print(f"Info: {info_sample}")
            trees.append((method, parsed))

        return trees, replace_labels

    def _build_rename_map(self, replace_labels: set[tuple[str, str]]) -> dict[str, str]:
        """Build a randomized rename mapping for user-defined variables."""
        types: dict[str, int] = {}
        for _, t in replace_labels:
            types[t] = types.get(t, 0) + 1

        if any(count > self.id_range for count in types.values()):
            print(
                f"Warning: User defined types {types} exceed id_range {self.id_range}. "
                "This may lead to ood tokens in the tokenizer."
            )

        draw_from = {
            k: random.sample(range(max(v, self.id_range)), k=v)
            for k, v in types.items()
        }
        return {name: f"{t}.{draw_from[t].pop(0)}" for name, t in replace_labels}

    def _trees_to_strings(
        self,
        trees: list[tuple[str, list[FOLTree | str]]],
        rename: dict[str, str] | None,
    ) -> list[tuple[str, tuple[str, ...]]]:
        """Convert parsed FOLTrees to string representations."""
        result: list[tuple[str, tuple[str, ...]]] = []
        for method, method_trees in trees:
            strs = []
            for tree in method_trees:
                if isinstance(tree, FOLTree):
                    if rename:
                        tree.rename(rename)
                    strs.append(tree.to_str(key=self.str_key, indent=self.str_indent))
                else:
                    strs.append(tree)
            result.append((method, tuple(strs)))
        return result

    def get_proofmethods(
        self,
        pfm_batch: list[list[tuple[str, list[dict]]]],
    ) -> list[list[tuple[str, tuple[str, ...]]]]:
        results = []
        for sample in pfm_batch:
            trees, replace_labels = self._parse_sample(sample, None)
            rename = self._build_rename_map(replace_labels) if replace_labels else None
            results.append(self._trees_to_strings(trees, rename))
        return results

    def tokenize(
        self,
        pfm_batch: list[list[tuple[str, list[dict]]]],
        **kwargs: object,
    ) -> list[list[list[str]]]:
        proofmethods_batch = self.get_proofmethods(pfm_batch)
        sep_token = self.tokenizer.sep_token or "[SEP]"
        eos_token = self.tokenizer.eos_token or "[EOS]"

        if self.max_length is not None:
            kwargs["truncation"] = True
            kwargs["max_length"] = self.max_length - 1  # because we add an EOS token

        return [
            [
                self.tokenizer.tokenize(sep_token.join([method] + list(objs)), **kwargs)
                + [eos_token]
                for method, objs in sample
            ]
            for sample in proofmethods_batch
        ]

    def encode(
        self,
        pfm_batch: list[list[tuple[str, list[dict]]]],
        **kwargs: object,
    ) -> list[list[list[int]]]:
        tokenized_batch = self.tokenize(pfm_batch, **kwargs)
        return [
            [self.tokenizer.convert_tokens_to_ids(tokens) for tokens in sample]
            for sample in tokenized_batch
        ]

    def _to_batch(
        self,
        pfm: list[tuple[str, list]] | list[list[tuple[str, list]]],
    ) -> list[list[tuple[str, list]]]:
        if isinstance(pfm[0], tuple):
            return [pfm]  # type: ignore
        return pfm  # type: ignore

    def __call__(
        self,
        pfm: list[tuple[str, list]] | list[list[tuple[str, list]]],
        padding: bool = True,
        return_attention_mask: bool = True,
        return_tensors: str | None = None,
        **kwargs: object,
    ) -> tuple:
        if self.max_length is not None:
            kwargs["truncation"] = True
            kwargs["max_length"] = self.max_length - 1  # because we add an EOS token

        pfm_batch = self._to_batch(pfm)

        if self.replace_var in ("all", "user"):
            print(
                "Warning: The replace_var option is significantly slower "
                "and cannot use caching."
            )
            input_ids_batch = self.encode(pfm_batch, **kwargs)
        else:
            input_ids_batch = self._encode_fast(pfm_batch, **kwargs)

        # Flatten input_ids and record slice boundaries
        flat_input_ids: list[list[int]] = []
        slices: list[int] = [0]
        for sample in input_ids_batch:
            flat_input_ids.extend(sample)
            slices.append(len(flat_input_ids))

        batch = self.tokenizer.pad(
            {"input_ids": flat_input_ids},
            padding=padding,
            return_attention_mask=return_attention_mask,
            return_tensors=return_tensors,
        )
        if return_tensors == "pt":
            return batch, torch.tensor(slices, dtype=torch.long)
        return batch, slices

    def _encode_fast(
        self,
        pfm_batch: list[list[tuple[str, list]]],
        **kwargs: object,
    ) -> list[list[list[int]]]:
        """Fast path: batch tokenization with LRU caching."""
        all_pfms = [p for sample in pfm_batch for p in sample]
        results: list[list[int] | None] = [None] * len(all_pfms)
        miss_indices: list[int] = []

        if self.cache is not None:
            cache_keys = [make_hashable(p) for p in all_pfms]
            miss_pfms: list[tuple[str, list]] = []
            for i, (key, p) in enumerate(zip(cache_keys, all_pfms)):
                cached = self.cache.get(key)
                if cached:
                    results[i] = cached
                else:
                    miss_indices.append(i)
                    miss_pfms.append(p)
        else:
            miss_indices = list(range(len(all_pfms)))
            miss_pfms = all_pfms

        if miss_pfms:
            sep_token = self.tokenizer.sep_token or "[SEP]"
            eos_token = self.tokenizer.eos_token or "[EOS]"
            texts = [
                sep_token.join([method] + list(objs)) + eos_token
                for method, objs in miss_pfms
            ]
            tokenized_misses = self.tokenizer(texts, **kwargs)["input_ids"]

            for i, miss_idx in enumerate(miss_indices):
                results[miss_idx] = tokenized_misses[i]
                if self.cache is not None:
                    self.cache[cache_keys[miss_idx]] = tokenized_misses[i]

        # Reconstruct batch structure from flat results
        input_ids_batch: list[list[list[int]]] = []
        start = 0
        for sample in pfm_batch:
            end = start + len(sample)
            input_ids_batch.append(results[start:end])  # type: ignore
            start = end
        return input_ids_batch
