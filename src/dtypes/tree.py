"""Tree class inspired by the nltk tree class https://www.nltk.org/_modules/nltk/tree.html"""

import hashlib
import json
from typing import Callable, Generic, TypeVar

from typing_extensions import Self

T = TypeVar("T")
U = TypeVar("U")


class Tree(list, Generic[T]):
    def __init__(self, label: T, children: list["Tree[T]"]):
        list.__init__(self, children)
        self.label = label

    @property
    def leaves(self) -> list[T]:
        """Returns a list of (labels of) leaves"""
        if not self:
            return [self.label]
        leaves = []
        for child in self:
            leaves.extend(child.leaves)
        return leaves

    @property
    def breadth(self) -> int:
        """Returns the breadth of the tree, i.e., the number of leaves"""
        return len(self.leaves)

    @property
    def height(self) -> int:
        """Returns the height of the tree, i.e., the length of the longest path from the root node to a leaf node"""
        max_child_height = -1
        for child in self:
            max_child_height = max(max_child_height, child.height)
        return max_child_height + 1

    def size(self, **kwargs) -> int:
        """Returns the size of the tree, i.e., the number of nodes"""
        size = 1
        for child in self:
            size += child.size(**kwargs)
        return size

    def fold(self, f: Callable[[T, list[U]], U]) -> U:
        """Tree folding"""
        childs = [child.fold(f) for child in self]
        return f(self.label, childs)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (
                self.label == other.label
                and len(self) == len(other)
                and all(c1 == c2 for c1, c2 in zip(self, other))
            )
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self, depth=None) -> str:
        depth = 0 if depth is None else depth
        childs = ", ".join(child.__repr__(depth + 1) for child in self)
        if len(childs) > 0:
            indent = "\t" * depth
            return f"{type(self).__name__}@{id(self)}({repr(self.label)} [\n\t{indent}{childs}\n{indent}])"
        else:
            return f"{type(self).__name__}@{id(self)}({repr(self.label)})"


class MerkleTree(Tree[dict[str, str]]):

    def __init__(self, label: dict[str, str], children: list[Self]):
        super(MerkleTree, self).__init__(label, children)  # type: ignore
        self.hash = hashlib.sha256(json.dumps(label).encode("utf-8"))
        for child in children:
            self.hash.update(child.hash.digest())

    def rename(self, rename: dict[str, str]):
        if "name" in self.label and self.label["name"] in rename.keys():
            self.label["name"] = rename[self.label["name"]]
        if len(self) > 0:
            for child in self:
                child.rename(rename)

    def clean_heristic_names(self):
        if "name" in self.label and (
            self.label["name"].startswith("L_") or self.label["name"].startswith("F_")
        ):
            self.label["name"] = self.label["name"][2:]
        if len(self) > 0:
            for child in self:
                child.clean_heristic_names()

    @classmethod
    def from_hashed(
        cls,
        label: dict[str, str],
        children: list[Self],
        hashmap: dict[bytes, Self],
    ) -> Self:
        subtree_hash = hashlib.sha256(json.dumps(label).encode("utf-8"))
        for child in children:
            subtree_hash.update(child.hash.digest())
        if subtree_hash.digest() in hashmap:
            return hashmap[subtree_hash.digest()]
        self = cls(label, children)
        self.hash = subtree_hash
        hashmap[subtree_hash.digest()] = self
        return self

    @classmethod
    def from_expression(cls, json_ast: dict, hashmap: dict[bytes, Self]) -> Self:
        return cls.from_hashed(
            label={k: v for k, v in json_ast.items() if k != "args"},
            children=[
                cls.from_expression(child, hashmap) for child in json_ast["args"]
            ],
            hashmap=hashmap,
        )

    def to_str(
        self,
        key: None | str = None,
        max_depth=None,
        depth=None,
        indent: bool = True,
    ) -> str:
        depth = 0 if depth is None else depth
        if key is None:
            label = str(self.label)
        else:
            if key in self.label:
                label = str(self.label[key])
            else:
                raise ValueError(f"Key {key} not found in label")

        if max_depth is not None and depth >= max_depth - 1 and len(self) > 0:
            return f"{label}(...)"

        childs = ", ".join(
            child.to_str(key, max_depth, depth + 1, indent=indent) for child in self
        )
        if len(childs) == 0:
            return label

        indent_str = "\t" * depth if indent else ""
        single_ident = "\t" if indent else ""
        new_lines_str = "\n" if indent else ""
        return f"{label}({new_lines_str}{single_ident}{indent_str}{childs}{new_lines_str}{indent_str})"
