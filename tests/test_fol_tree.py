import pytest

from src.dtypes.fol_tree import FOLTree, rename_quantified
from tests.conftest import _leaf as leaf_ast
from tests.conftest import _node as node_ast

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def forall_ast(vars_, body_args):
    return {"name": "Forall", "type": "quant", "vars": vars_, "args": body_args}


# ---------------------------------------------------------------------------
# FOLTree — construction & node sharing
# ---------------------------------------------------------------------------


class TestFOLTreeConstruction:
    def test_leaf(self):
        t = FOLTree.from_expression(leaf_ast("x"), {})
        assert t.label["name"] == "x"
        assert len(t) == 0
        assert t.vars == []

    def test_nested(self):
        ast = node_ast("KU", [leaf_ast("x")])
        t = FOLTree.from_expression(ast, {})
        assert t.label["name"] == "KU"
        assert t[0].label["name"] == "x"

    def test_node_sharing(self):
        child = leaf_ast("x")
        ast = node_ast("pair", [child, child])
        t = FOLTree.from_expression(ast, {})
        assert t[0] is t[1]

    def test_shared_hashmap_reuses(self):
        hashmap = {}
        t1 = FOLTree.from_expression(leaf_ast("x"), hashmap)
        t2 = FOLTree.from_expression(leaf_ast("x"), hashmap)
        assert t1 is t2


# ---------------------------------------------------------------------------
# FOLTree — quantifiers
# ---------------------------------------------------------------------------


class TestFOLTreeQuantifiers:
    def test_forall_binds_vars(self):
        ast = forall_ast(vars_=[leaf_ast("x")], body_args=[leaf_ast("x")])
        t = FOLTree.from_expression(ast, {})
        assert len(t.vars) == 1
        assert t[0] is t.vars[0]

    def test_multiple_quantified_vars(self):
        ast = forall_ast(
            vars_=[leaf_ast("x"), leaf_ast("y")],
            body_args=[node_ast("pair", [leaf_ast("x"), leaf_ast("y")])],
        )
        t = FOLTree.from_expression(ast, {})
        assert len(t.vars) == 2
        assert t[0][0] is t.vars[0]
        assert t[0][1] is t.vars[1]

    def test_nested_quantifiers(self):
        inner = forall_ast(vars_=[leaf_ast("y")], body_args=[leaf_ast("y")])
        outer = forall_ast(
            vars_=[leaf_ast("x")],
            body_args=[node_ast("pair", [leaf_ast("x"), inner])],
        )
        t = FOLTree.from_expression(outer, {})
        inner_tree = t[0][1]
        assert inner_tree.label["name"] == "Forall"
        assert len(inner_tree.vars) == 1


# ---------------------------------------------------------------------------
# FOLTree — commutative marking
# ---------------------------------------------------------------------------


class TestFOLTreeCommutative:
    def test_disj_and_conj_marked(self):
        for name in ("disj", "conj"):
            ast = node_ast(name, [leaf_ast("a"), leaf_ast("b")])
            t = FOLTree.from_expression(ast, {})
            assert t.label.get("commutative") == "True"

    def test_other_not_marked(self):
        t = FOLTree.from_expression(node_ast("KU", [leaf_ast("x")]), {})
        assert "commutative" not in t.label


# ---------------------------------------------------------------------------
# FOLTree — user_defined & labels
# ---------------------------------------------------------------------------


class TestFOLTreeLabels:
    def test_no_user_defined(self):
        t = FOLTree.from_expression(node_ast("KU", [leaf_ast("x")]), {})
        assert t.user_defined == set()

    def test_user_defined_collected(self):
        ast = node_ast(
            "pair",
            [
                leaf_ast("a", "msg", userDefined="True"),
                leaf_ast("b", "node", userDefined="True"),
            ],
        )
        t = FOLTree.from_expression(ast, {})
        assert t.user_defined == {("a", "msg"), ("b", "node")}

    def test_labels_collects_all(self):
        ast = node_ast("KU", [leaf_ast("x", "msg")], type_="fun")
        t = FOLTree.from_expression(ast, {})
        assert ("KU", "fun") in t.labels
        assert ("x", "msg") in t.labels


# ---------------------------------------------------------------------------
# rename_quantified
# ---------------------------------------------------------------------------


class TestRenameQuantified:
    def test_no_quantifiers(self):
        ast = node_ast("KU", [leaf_ast("x")])
        assert rename_quantified(ast) == 0

    def test_single_forall(self):
        body = leaf_ast("x")
        ast = forall_ast(vars_=[leaf_ast("x")], body_args=[body])
        assert rename_quantified(ast) == 1
        assert ast["vars"][0]["name"] == "VAR0"
        assert body["name"] == "VAR0"

    def test_nested_quantifiers(self):
        inner = forall_ast(vars_=[leaf_ast("y")], body_args=[leaf_ast("y")])
        outer = forall_ast(
            vars_=[leaf_ast("x")],
            body_args=[node_ast("pair", [leaf_ast("x"), inner])],
        )
        rename_quantified(outer)
        assert inner["vars"][0]["name"] == "VAR0"
        assert outer["vars"][0]["name"] == "VAR1"

    def test_var_prefix_rejected(self):
        with pytest.raises(ValueError, match="not allowed"):
            rename_quantified(leaf_ast("VAR0"))

    def test_alpha_equivalence(self):
        """Forall x. P(x) and Forall y. P(y) produce same VAR names."""
        ast1 = forall_ast(
            vars_=[leaf_ast("x")], body_args=[node_ast("P", [leaf_ast("x")])]
        )
        ast2 = forall_ast(
            vars_=[leaf_ast("y")], body_args=[node_ast("P", [leaf_ast("y")])]
        )
        rename_quantified(ast1)
        rename_quantified(ast2)
        assert ast1["vars"][0]["name"] == ast2["vars"][0]["name"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestFOLTreeEdgeCases:
    def test_deeply_nested(self):
        ast = leaf_ast("x")
        for _ in range(10):
            ast = node_ast("f", [ast])
        t = FOLTree.from_expression(ast, {})
        assert t.height == 10
