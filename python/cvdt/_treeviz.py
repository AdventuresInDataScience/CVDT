"""Render a CVDT tree (as returned by the native ``export_tree``) to text or
Graphviz DOT.

CVDT splits are membership tests, not CART-style thresholds: a continuous split
routes a sample *left* when its value falls in a half-open interval
``[lower, upper)`` (open ends shown as one-sided inequalities), and a categorical
split routes left when the feature equals a specific category. "Left" is always
the branch where the condition is **true**; anything else (including missing /
non-finite values) routes right.
"""

from __future__ import annotations

import math


def _feature_label(f, feature_names):
    if feature_names is not None and 0 <= f < len(feature_names):
        return str(feature_names[f])
    return f"x[{f}]"


def _num(v):
    return f"{v:.4g}"


def _condition(tree, i, feature_names):
    """Human-readable 'condition true' (left branch) for internal node i."""
    f = tree["feature"][i]
    name = _feature_label(f, feature_names)
    if tree["is_categorical"][i]:
        return f"{name} == {tree['category'][i]}"
    lo = tree["lower"][i]
    hi = tree["upper"][i]
    lo_open = lo is None or math.isinf(lo) or math.isnan(lo)
    hi_open = hi is None or math.isinf(hi) or math.isnan(hi)
    if lo_open and not hi_open:
        return f"{name} < {_num(hi)}"
    if hi_open and not lo_open:
        return f"{name} >= {_num(lo)}"
    if not lo_open and not hi_open:
        return f"{_num(lo)} <= {name} < {_num(hi)}"
    return f"{name} (any)"


def _leaf_label(tree, i, class_names):
    n = tree["n_samples"][i]
    if "predicted_class" in tree:
        c = tree["predicted_class"][i]
        if class_names is not None and 0 <= c < len(class_names):
            cname = str(class_names[c])
        else:
            cname = str(c)
        proba = tree.get("proba", [None] * tree["node_count"])[i]
        if proba:
            probs = "[" + ", ".join(f"{p:.3f}" for p in proba) + "]"
            return f"class={cname} n={n} proba={probs}"
        return f"class={cname} n={n}"
    return f"value={_num(tree['value'][i])} n={n}"


def export_text(tree, feature_names=None, class_names=None, indent="  "):
    """Return a readable, rule-style text rendering of the tree."""
    if tree.get("node_count", 0) == 0:
        return "(empty tree)"
    lines = []

    def recurse(i, depth):
        pad = indent * depth
        if tree["is_leaf"][i]:
            lines.append(f"{pad}{_leaf_label(tree, i, class_names)}")
            return
        cond = _condition(tree, i, feature_names)
        lines.append(f"{pad}if {cond}:")
        recurse(tree["children_left"][i], depth + 1)
        lines.append(f"{pad}else:  # not ({cond})")
        recurse(tree["children_right"][i], depth + 1)

    recurse(0, 0)
    return "\n".join(lines)


def _dot_escape(s):
    return s.replace("\\", "\\\\").replace('"', '\\"')


def export_graphviz(tree, feature_names=None, class_names=None):
    """Return a Graphviz DOT string (render with graphviz / pydot / dtreeviz)."""
    out = [
        "digraph CVDT {",
        '  node [shape=box, style="rounded,filled", fontname=helvetica, fillcolor="#f5f5f5"] ;',
        "  edge [fontname=helvetica] ;",
    ]
    nc = tree.get("node_count", 0)
    if nc == 0:
        out.append('  empty [label="(empty tree)"] ;')
        out.append("}")
        return "\n".join(out)

    for i in range(nc):
        if tree["is_leaf"][i]:
            label = _leaf_label(tree, i, class_names)
            out.append(f'  {i} [label="{_dot_escape(label)}", fillcolor="#e8f4ea"] ;')
        else:
            label = _condition(tree, i, feature_names) + f"\\nn={tree['n_samples'][i]}"
            out.append(f'  {i} [label="{_dot_escape(label)}"] ;')
    for i in range(nc):
        if not tree["is_leaf"][i]:
            lft = tree["children_left"][i]
            rgt = tree["children_right"][i]
            out.append(f'  {i} -> {lft} [label="true"] ;')
            out.append(f'  {i} -> {rgt} [label="false"] ;')
    out.append("}")
    return "\n".join(out)
