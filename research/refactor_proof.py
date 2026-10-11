"""Prove a refactor changes no meaning: snapshot what every name in a set of modules MEANS, before
and after, and compare.

WHY. A pure move -- a constant to the library that owns it, a function out of one study into
another, an alias renamed -- changes the TEXT of every module it touches, so a textual diff cannot
show that nothing changed, and tests only sample behaviour. What a name means is what its value
is and, for a function, what it computes with: its body with every global it reads resolved to the
object it reaches. That is what this records.

    python -m research.refactor_proof snapshot OUT.json PATTERN... [--root DIR] [--follow PREFIX]
    python -m research.refactor_proof compare BEFORE.json AFTER.json [--ignore NAME ...]

PATTERN is a dotted module name, or a glob over module names under the root
("research.experiments.memory_*", "research.experiments.memory_lib.*"). For every module-level
name (and every method and property of a class defined there) the snapshot records:

    a constant       its repr; containers element-wise; the root's own path replaced by <ROOT>
    a module         its name
    a class          where it is defined; and each of its methods and properties --
                     its own, and those of bases under --follow, resolved as the class
                     resolves them, so a method moved into a mixin keeps its entry
    a function       its IDENTITY: the hash of its body -- docstring and name removed -- with every
                     global it reads (a bare name, alias.attr, or a name a function-local import
                     binds) replaced by what that resolves to: constants by repr, modules by name,
                     functions under --follow (default research.experiments) by their own
                     identity, recursively; anything else by its qualified name

so a function moved to another module, or reached through another alias, or imported from
another place, keeps its identity exactly when it computes with the same objects; changing a
constant it reads, or a function it calls, changes it. ``compare`` lists the names whose meaning
differs, the names that appeared and the names that disappeared.

The BEFORE snapshot is taken of the commit itself, in a worktree beside the repository, with the
same tool, in a separate process:

    git worktree add --detach ../assemblies-proof-head HEAD
    python -m research.refactor_proof snapshot before.json PATTERN... --root ../assemblies-proof-head
    python -m research.refactor_proof snapshot after.json PATTERN...
    python -m research.refactor_proof compare before.json after.json

Limits: what a function does through an object it is PASSED (not a global) is not followed; two
functions with identical bodies are identical. A refactor that rewrites a function's body (not
only what it reaches) needs its own argument -- tests, research.replay.
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import importlib
import inspect
import json
import pathlib
import sys
import textwrap
import types

SIMPLE = (int, float, complex, str, bytes, bool, type(None), range)


class Meaning:
    """The meaning of names under one root, following functions whose module starts with
    ``follow``."""

    def __init__(self, root, follow="research.experiments"):
        self.root = str(pathlib.Path(root).resolve())
        self.follow = follow
        self._memo: dict[int, str] = {}
        self._active: set[int] = set()

    # -------------------------------------------------------------- tokens
    def token(self, obj):
        if isinstance(obj, types.ModuleType):
            return f"module:{obj.__name__}"
        if isinstance(obj, types.FunctionType):
            if not (obj.__module__ or "").startswith(self.follow):
                return f"fn:ext:{obj.__module__}.{obj.__qualname__}"
            return f"fn:{self.identity(obj)}"
        if isinstance(obj, (set, frozenset)):
            return f"{type(obj).__name__}[" + ",".join(sorted(self.token(x) for x in obj)) + "]"
        if isinstance(obj, (tuple, list)):
            return f"{type(obj).__name__}(" + ",".join(self.token(x) for x in obj) + ")"
        if isinstance(obj, dict):
            return "dict{" + ",".join(f"{self.token(k)}:{self.token(v)}" for k, v in obj.items()) + "}"
        if isinstance(obj, type):
            if not (obj.__module__ or "").startswith(self.follow):
                return f"class:{obj.__module__}.{obj.__qualname__}"
            return f"class:{obj.__qualname__}:{self.class_identity(obj)}"
        if isinstance(obj, str):
            for r in {self.root, self.root.replace("\\", "/"), self.root.replace("/", "\\")}:
                obj = obj.replace(r, "<ROOT>")
        if isinstance(obj, SIMPLE):
            return "const:" + repr(obj)
        if callable(obj) and hasattr(obj, "__module__") and hasattr(obj, "__qualname__"):
            return f"callable:{obj.__module__}.{obj.__qualname__}"
        if (type(obj).__module__ or "").startswith(self.follow):
            return f"obj:{self.token(type(obj))}"           # an instance of a followed class
        return f"obj:{type(obj).__module__}.{type(obj).__qualname__}"

    # -------------------------------------------------------------- classes
    def class_identity(self, cls):
        """a followed class by what it is, not where it lives: its bases and every member it
        defines (methods by identity, other attributes by token); a class moved to another
        module keeps it"""
        key = id(cls)
        if key in self._memo:
            return self._memo[key]
        if key in self._active:
            return f"cycle:{cls.__qualname__}"
        self._active.add(key)
        parts = [",".join(b.__qualname__ for b in cls.__bases__)]
        for k, v in sorted(vars(cls).items()):
            if k in ("__module__", "__dict__", "__weakref__", "__doc__", "__qualname__"):
                continue
            if isinstance(v, (staticmethod, classmethod)):
                v = v.__func__
            if isinstance(v, property):
                v = v.fget
            parts.append(f"{k}={self.token(v)}")
        h = _sha("|".join(parts))
        self._active.discard(key)
        self._memo[key] = h
        return h

    # -------------------------------------------------------------- functions
    def identity(self, fn):
        key = id(fn)
        if key in self._memo:
            return self._memo[key]
        if key in self._active:
            return f"cycle:{fn.__name__}"
        if fn.__name__ == "<lambda>":                    # its source is a fragment of a line
            c = fn.__code__
            names = [self.token(fn.__globals__.get(n)) for n in c.co_names]
            h = "lambda:" + _sha(repr((c.co_code, c.co_consts, c.co_names, c.co_varnames, names)))
            self._memo[key] = h
            return h
        self._active.add(key)
        try:
            node = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                raise TypeError("not a def")
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
            node.name = "_"
            bound = _local_imports(node)
            local = _locals(node) - set(bound)
            node = _Resolve(self, {**fn.__globals__, **bound}, local).visit(node)
            h = _sha(ast.dump(node, include_attributes=False))
        except (OSError, TypeError):
            h = f"nosource:{fn.__qualname__}"      # generated (e.g. a dataclass __init__)
        self._active.discard(key)
        self._memo[key] = h
        return h

    # -------------------------------------------------------------- modules
    def module(self, name):
        mod = importlib.import_module(name)
        entry = {}
        for k, v in sorted(vars(mod).items()):
            if k.startswith("__"):
                continue
            entry[k] = self.token(v)
            # a class defined here, or a followed class re-exported here (a facade)
            if isinstance(v, type) and (v.__module__ == mod.__name__
                                        or (v.__module__ or "").startswith(self.follow)):
                # every member the class resolves, its own or a followed base's: a method moved
                # into a mixin keeps its key and, computing the same, its identity
                members = {}
                for base in reversed(v.__mro__):
                    if base is v or (base.__module__ or "").startswith(self.follow):
                        members.update(vars(base))
                for mk, mv in sorted(members.items()):
                    if isinstance(mv, types.FunctionType):
                        entry[f"{k}.{mk}"] = "fn:" + self.identity(mv)
                    elif isinstance(mv, (staticmethod, classmethod)) and isinstance(mv.__func__, types.FunctionType):
                        entry[f"{k}.{mk}"] = "fn:" + self.identity(mv.__func__)
                    elif isinstance(mv, property) and isinstance(mv.fget, types.FunctionType):
                        entry[f"{k}.{mk}"] = "property:" + self.identity(mv.fget)
        return entry


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _locals(fn_node):
    names = set()
    for n in ast.walk(fn_node):
        if isinstance(n, ast.arg):
            names.add(n.arg)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            names.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n is not fn_node:
            names.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in n.names)
    return names


def _local_imports(fn_node):
    """name -> object, for every absolute import inside the function"""
    out = {}
    for n in ast.walk(fn_node):
        if isinstance(n, ast.ImportFrom) and not n.level and n.module:
            mod = importlib.import_module(n.module)
            for a in n.names:
                out[a.asname or a.name] = (getattr(mod, a.name) if hasattr(mod, a.name)
                                           else importlib.import_module(f"{n.module}.{a.name}"))
        elif isinstance(n, ast.Import):
            for a in n.names:
                importlib.import_module(a.name)
                top = a.name.split(".")[0]
                out[a.asname or top] = importlib.import_module(a.name if a.asname else top)
    return out


class _Resolve(ast.NodeTransformer):
    """every global a body reads, replaced by the token of what it resolves to"""

    def __init__(self, meaning, glb, local):
        self.m, self.glb, self.local = meaning, glb, local

    def visit_ImportFrom(self, node):
        # an absolute function-local import binds names resolved through glb; where it imports
        # them from is not meaning
        return node if node.level else None

    def visit_Import(self, node):
        return None

    def visit_Attribute(self, node):
        if isinstance(node.ctx, ast.Load):
            parts, base = [], node
            while isinstance(base, ast.Attribute):
                parts.append(base.attr)
                base = base.value
            if isinstance(base, ast.Name) and base.id not in self.local and base.id in self.glb:
                obj = self.glb[base.id]
                for a in reversed(parts):
                    if not isinstance(obj, types.ModuleType) or not hasattr(obj, a):
                        break
                    obj = getattr(obj, a)
                else:
                    return ast.copy_location(ast.Constant(value=self.m.token(obj)), node)
        self.generic_visit(node)
        return node

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id not in self.local and node.id in self.glb:
            return ast.copy_location(ast.Constant(value=self.m.token(self.glb[node.id])), node)
        return node


# ------------------------------------------------------------------ module sets
def expand(patterns, root):
    """dotted module names matching each pattern, found under ``root``"""
    root = pathlib.Path(root)
    found = []
    for pat in patterns:
        if not any(c in pat for c in "*?["):
            found.append(pat)
            continue
        base = pat.split("*")[0].rsplit(".", 1)[0]
        here = root.joinpath(*base.split("."))
        for p in sorted(here.rglob("*.py")):
            rel = p.relative_to(root).with_suffix("")
            parts = rel.parts[:-1] if rel.name == "__init__" else rel.parts
            name = ".".join(parts)
            if fnmatch.fnmatchcase(name, pat) and name not in found:
                found.append(name)
    return found


def snapshot(patterns, root=".", follow="research.experiments"):
    root = str(pathlib.Path(root).resolve())
    sys.path.insert(0, root)
    meaning = Meaning(root, follow)
    return {name: meaning.module(name) for name in expand(patterns, root)}


def compare(before, after, ignore=()):
    """{"changed": [(module, name, before, after)], "added": [...], "removed": [...]}"""
    out = {"changed": [], "added": [], "removed": []}
    for m in sorted(set(before) | set(after)):
        b, a = before.get(m, {}), after.get(m, {})
        for k in sorted(set(a) | set(b)):
            if k in ignore or b.get(k) == a.get(k):
                continue
            kind = "changed" if k in a and k in b else ("added" if k in a else "removed")
            out[kind].append((m, k, b.get(k), a.get(k)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("out")
    s.add_argument("patterns", nargs="+")
    s.add_argument("--root", default=".")
    s.add_argument("--follow", default="research.experiments")
    c = sub.add_parser("compare")
    c.add_argument("before")
    c.add_argument("after")
    c.add_argument("--ignore", nargs="*", default=[])
    args = ap.parse_args(argv)
    if args.command == "snapshot":
        snap = snapshot(args.patterns, args.root, args.follow)
        pathlib.Path(args.out).write_text(json.dumps(snap, indent=1, sort_keys=True), encoding="utf-8")
        print(f"{len(snap)} modules, {sum(len(v) for v in snap.values())} names")
        return 0
    result = compare(json.loads(pathlib.Path(args.before).read_text(encoding="utf-8")),
                     json.loads(pathlib.Path(args.after).read_text(encoding="utf-8")), set(args.ignore))
    for kind in ("changed", "removed", "added"):
        print(f"{kind}: {len(result[kind])}")
        for m, k, b, a in result[kind]:
            print(f"  {m}  {k}  {b} -> {a}")
    return 1 if result["changed"] else 0


if __name__ == "__main__":
    sys.exit(main())
