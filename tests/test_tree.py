import pytest

from src.dtypes.tree import MerkleTree, Tree

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def leaf(label):
    return Tree(label, [])


def tree_abc():
    """Tree: a(b, c)"""
    return Tree("a", [leaf("b"), leaf("c")])


def deep_tree():
    """Tree: a(b(c(d)))"""
    return Tree("a", [Tree("b", [Tree("c", [leaf("d")])])])


def mleaf(name, type_="node"):
    return MerkleTree({"name": name, "type": type_}, [])


def mtree_ku():
    """MerkleTree: KU(x)"""
    return MerkleTree({"name": "KU", "type": "fun"}, [mleaf("x")])


# ---------------------------------------------------------------------------
# Tree — construction & properties
# ---------------------------------------------------------------------------


class TestTree:
    def test_leaf(self):
        t = leaf("x")
        assert t.label == "x"
        assert len(t) == 0

    def test_children(self):
        t = tree_abc()
        assert t.label == "a"
        assert len(t) == 2
        assert t[0].label == "b"
        assert t[1].label == "c"

    def test_leaves(self):
        assert leaf("x").leaves == ["x"]
        assert set(tree_abc().leaves) == {"b", "c"}
        assert deep_tree().leaves == ["d"]

    def test_height(self):
        assert leaf("x").height == 0
        assert tree_abc().height == 1
        assert deep_tree().height == 3

    def test_size(self):
        assert leaf("x").size() == 1
        assert tree_abc().size() == 3
        assert deep_tree().size() == 4

    def test_fold(self):
        assert leaf(5).fold(lambda label, children: label) == 5
        t = Tree(1, [Tree(2, [leaf(3)]), leaf(4)])
        assert t.fold(lambda label, children: label + sum(children)) == 10

    def test_equality(self):
        assert tree_abc() == tree_abc()
        assert tree_abc() != Tree("z", [leaf("b"), leaf("c")])
        assert tree_abc() != Tree("a", [leaf("b")])
        assert tree_abc() != "not a tree"

    def test_repr(self):
        assert "'x'" in repr(leaf("x"))
        assert "'a'" in repr(tree_abc())


# ---------------------------------------------------------------------------
# MerkleTree — hashing & construction
# ---------------------------------------------------------------------------


class TestMerkleTree:
    def test_construction(self):
        t = mleaf("x")
        assert t.label == {"name": "x", "type": "node"}
        assert t.hash is not None

    def test_hash_deterministic(self):
        assert mleaf("x").hash.digest() == mleaf("x").hash.digest()
        assert mleaf("x").hash.digest() != mleaf("y").hash.digest()

    def test_children_affect_hash(self):
        t1 = MerkleTree({"name": "f"}, [mleaf("x")])
        t2 = MerkleTree({"name": "f"}, [mleaf("y")])
        assert t1.hash.digest() != t2.hash.digest()

    def test_from_expression(self):
        ast = {
            "name": "KU",
            "type": "fun",
            "args": [{"name": "x", "type": "node", "args": []}],
        }
        t = MerkleTree.from_expression(ast, {})
        assert t.label == {"name": "KU", "type": "fun"}
        assert t[0].label == {"name": "x", "type": "node"}

    def test_node_sharing(self):
        child_ast = {"name": "x", "type": "node", "args": []}
        ast = {"name": "pair", "type": "fun", "args": [child_ast, child_ast]}
        t = MerkleTree.from_expression(ast, {})
        assert t[0] is t[1]

    def test_rename(self):
        t = mtree_ku()
        t.rename({"x": "z"})
        assert t[0].label["name"] == "z"
        assert t.label["name"] == "KU"

    def test_clean_heristic_names(self):
        inner = mleaf("L_x")
        t = MerkleTree({"name": "F_outer", "type": "fun"}, [inner])
        t.clean_heristic_names()
        assert t.label["name"] == "outer"
        assert t[0].label["name"] == "x"

    def test_to_str(self):
        assert mleaf("x").to_str(key="name") == "x"
        assert mtree_ku().to_str(key="name", indent=False) == "KU(x)"
        assert mtree_ku().to_str(key="name", max_depth=1, indent=False) == "KU(...)"

    def test_to_str_invalid_key(self):
        with pytest.raises(ValueError, match="Key .* not found"):
            mleaf("x").to_str(key="nonexistent")
