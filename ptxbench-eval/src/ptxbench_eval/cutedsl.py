"""CuTeDSL source reference provenance and CPU-only submission preflight."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any


def source_snapshot(root: Path) -> dict[str, Any]:
    """Validate an exported installed-source snapshot before mounting it."""
    manifest_path = root / "source_manifest.json"
    encoded = manifest_path.read_bytes()
    manifest = json.loads(encoded)
    files = manifest.get("files")
    version = manifest.get("version")
    if (manifest.get("schema_version") != 1 or not isinstance(version, str)
            or not version or not isinstance(files, dict)
            or "cutlass/cute/__init__.py" not in files
            or manifest.get("packages", {}).get("nvidia-cutlass-dsl") != version):
        raise ValueError("invalid CuTeDSL source_manifest.json")
    actual = set()
    total_bytes = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"CuTeDSL source contains a symlink: {path}")
        if not path.is_file() or path == manifest_path:
            continue
        relative = path.relative_to(root).as_posix()
        actual.add(relative)
        content = path.read_bytes()
        total_bytes += len(content)
        if hashlib.sha256(content).hexdigest() != files.get(relative):
            raise ValueError(f"CuTeDSL source hash mismatch: {relative}")
    if actual != set(files):
        raise ValueError("CuTeDSL source file set differs from source_manifest.json")
    return {
        "source": str(root.resolve()),
        "workspace_path": "cutedsl",
        "version": version,
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "file_count": len(files),
        "total_bytes": total_bytes,
    }


def validation_error(source: str) -> str | None:
    """Check Python without importing or executing the submitted module."""
    try:
        tree = ast.parse(source, filename="kernel.py")
        compile(tree, "kernel.py", "exec")
    except (SyntaxError, ValueError) as exc:
        return f"ERROR: Invalid Python in kernel.py: {exc}"
    if not any(isinstance(node, ast.FunctionDef) and node.name == "run" for node in tree.body):
        return "ERROR: kernel.py requires a destination-passing run function."
    # Resolve import aliases for common library shortcuts. This is a preflight,
    # not a sandbox or a proof of the submitted kernel's implementation.
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                aliases[name.asname or name.name.split('.')[0]] = name.name if name.asname else name.name.split('.')[0]
        elif isinstance(node, ast.ImportFrom):
            for name in node.names:
                aliases[name.asname or name.name] = f"{node.module}.{name.name}"

    def qualified(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return f"{qualified(node.value)}.{node.attr}"
        return ""

    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            return "ERROR: Implement GPU computation in CuTeDSL; Python matrix multiplication is not allowed."
        if isinstance(node, ast.Call):
            name = qualified(node.func)
            if (name in {"torch.mm", "torch.matmul", "torch.bmm", "torch.addmm", "torch.einsum"}
                    or "cublas" in name.lower() or "cudnn" in name.lower()):
                return f"ERROR: Implement GPU computation in CuTeDSL; library shortcut {name} is not allowed."
    return None
