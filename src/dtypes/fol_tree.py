"""Tree class inspired by the nltk tree class https://www.nltk.org/_modules/nltk/tree.html"""

import hashlib
import json
from copy import deepcopy
from typing import Any

from typing_extensions import Self

from .tree import MerkleTree


class FOLTree(MerkleTree):
    """First Order Logic Tree with node sharing"""

    def __init__(
        self,
        label: dict[str, str],
        children: list[Self],
    ):
        super(FOLTree, self).__init__(label, children)  # type: ignore
        self.hash = hashlib.sha256(json.dumps(label).encode("utf-8"))
        # quantified variables
        self.vars: list[Self] = []
        for child in children:
            self.hash.update(child.hash.digest())

    @classmethod
    def from_hashed(
        cls,
        label: dict[str, str],
        children: list[Self],
        hashmap: dict[bytes, Self],
    ) -> Self:
        return cls._from_hashed(label, children, {}, hashmap)

    @classmethod
    def from_expression(cls, json_ast, hashmap: dict[bytes, Self]) -> Self:
        json_ast = deepcopy(json_ast)
        rename_quantified(json_ast)
        return cls._from_expression(json_ast, hashmap, {})

    @classmethod
    def _from_hashed(
        cls,
        label: dict[str, str],
        children: list[Self],
        local_hashmap: dict[bytes, Self],
        global_hashmap: dict[bytes, Self],
    ) -> Self:
        subtree_hash = hashlib.sha256(json.dumps(label).encode("utf-8"))

        # if is leaf, look up in the local hashmap as this could be a quantified var
        if len(children) == 0 and subtree_hash.digest() in local_hashmap:
            return local_hashmap[subtree_hash.digest()]

        for child in children:
            subtree_hash.update(child.hash.digest())

        if subtree_hash.digest() in global_hashmap:
            return global_hashmap[subtree_hash.digest()]

        self = cls(label, children)  # type: ignore
        self.hash = subtree_hash
        global_hashmap[subtree_hash.digest()] = self
        local_hashmap[subtree_hash.digest()] = self

        return self

    @classmethod
    def _from_expression(
        cls,
        json_ast: dict[str, Any],
        global_hashmap: dict[bytes, Self],
        local_hashmap: dict[bytes, Self],
    ) -> Self:
        """Create a quantifier scope hashmap that contains only quantified
        variables. Those are not unique but don't node share yet.
        When recursing to a leaf we first look up in the quantifier scope hashmap
        and then in the global hashmap. Assumes unique names inside each scope.
        When going back, hence having the subtree constructed and back at the
        quantifier level, we check if the whole quantifier is in the global hashmap.
        If not, we add it to the global hashmap. That means, no quantified leafs are
        in the global hashmap. It also means that
        id(tree1) == id(tree2) <-/-> hash(tree1) == hash(tree2).
        Hash is unique for syntactically equal trees, id is not, only for those that
        we actually want to node share."""

        if json_ast["name"] == "Forall" or json_ast["name"] == "Exists":
            # quantifier node
            quantified: dict[bytes, Self] = {}
            for q_var in json_ast["vars"]:

                cls._from_hashed(
                    label={k: v for k, v in q_var.items() if k != "args"},
                    children=[],
                    local_hashmap=quantified,
                    global_hashmap={},
                )

            json_ast = {k: v for k, v in json_ast.items() if k != "vars"}
            local_hashmap = local_hashmap.copy()
            local_hashmap.update(quantified)

        children = []
        for arg in json_ast["args"]:
            child = cls._from_expression(
                arg,
                global_hashmap,
                local_hashmap,
            )
            children.append(child)

        json_ast = {k: str(v) for k, v in json_ast.items() if k != "args"}

        if json_ast["name"] == "disj" or json_ast["name"] == "conj":
            # commutative operator
            json_ast["commutative"] = "True"

        self = cls._from_hashed(
            label=json_ast,
            children=children,
            local_hashmap=local_hashmap,
            global_hashmap=global_hashmap,
        )

        if json_ast["name"] == "Forall" or json_ast["name"] == "Exists":
            self.vars = list(quantified.values())

        return self

    @property
    def user_defined(self) -> set[tuple[str, str]]:
        """Returns a set of user-defined nodes in the tree."""
        user_defined_types = set()
        for child in self:
            user_defined_types.update(child.user_defined)
        if "userDefined" in self.label and self.label["userDefined"]:
            user_defined_types.add((self.label["name"], self.label["type"]))
        return user_defined_types

    @property
    def labels(self) -> set[tuple[str, str]]:
        """Returns a set of user-defined nodes in the tree."""
        label_types = set()
        for child in self:
            label_types.update(child.labels)
        label_types.add((self.label["name"], self.label["type"]))
        return label_types


def rename_quantified(f) -> int:
    """De Brujin index inspired renaming of quantified variables. Innermost quantified
    variables are named VAR0, VAR1, ... VARn, where n is the quantifier nesting counted
    from the inside out. Precisely, for a quantifier q that defines variable v, v is
    renamed to VARi where i is the maximum depth of quantified variable nesting in the
    scope of the q. For multiple variables defined by the same quantifier, the variables
    are renamed to VARi, VARi+1, ... VARi+n where the order is given by which variables
    occurs first in the traversal. This allows node sharing between formulas that are
    identical but differently named. Example: Forall x. P(x) and Forall y. P(y)
    will both be renamed to Forall VAR0. P(VAR0).
    """

    nesting = max((rename_quantified(child) for child in f["args"]), default=0)

    if f["name"].startswith("VAR"):
        raise ValueError(
            f"Variable {f['name']} is not allowed in the input. Please use sth. that doesn't start with VAR."
        )

    if f["name"] == "Exists" or f["name"] == "Forall":
        quantifiers: dict[str, int | None] = {var["name"]: None for var in f["vars"]}
        rename_quantified_inner(f, nesting, quantifiers)
        assert sum(1 for v in quantifiers.values() if v is not None) == len(f["vars"])
        for var in f["vars"]:
            var["name"] = "VAR" + str(nesting + quantifiers[var["name"]])  # type: ignore
        nesting = nesting + len(f["vars"])

    return nesting


def rename_quantified_inner(f, nesting, quantifiers: dict[str, int | None]):
    if f["name"] in quantifiers:
        assert len(f["args"]) == 0
        quantifier = f["name"]
        if quantifiers[quantifier] is None:
            quantifiers[quantifier] = sum(
                1 for v in quantifiers.values() if v is not None
            )
        f["name"] = "VAR" + str(nesting + quantifiers[quantifier])
    else:
        for child in f["args"]:
            rename_quantified_inner(child, nesting, quantifiers)
