"""A small JSON Schema validator (standard library only).

Used in two places: tool arguments are checked against the tool's
``input_schema`` before the tool runs, and a model's structured output is
checked against ``AgentSpec.output_schema`` before it is returned. The point
is to tell the model precisely what was wrong, so it can fix the call.

Supported keywords: ``type`` (a name or a list, with ``integer`` excluding
booleans), ``enum``, ``const``, ``properties``, ``required``,
``additionalProperties`` (bool or schema), ``items``, ``minItems``,
``maxItems``, ``minLength``, ``maxLength``, ``pattern``, ``minimum``,
``maximum``, ``exclusiveMinimum``, ``exclusiveMaximum``, ``anyOf``, ``oneOf``,
``allOf``, ``not`` and local ``$ref`` (``#/$defs/...`` or ``#/definitions/...``).
Anything else is ignored, so a schema a provider accepts is never rejected here.
"""

from __future__ import annotations

import json
import re
from typing import Any

_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
    "null": (type(None),),
}


def _is_type(value: Any, name: str) -> bool:
    if name in ("number", "integer") and isinstance(value, bool):
        return False
    if name == "integer" and isinstance(value, float):
        return value.is_integer()
    return isinstance(value, _TYPES.get(name, ()))


def _show(value: Any) -> str:
    s = json.dumps(value, default=str, ensure_ascii=False)
    return s if len(s) <= 60 else s[:57] + "..."


def _resolve(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    ref = schema.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return schema
    node: Any = root
    for part in ref[2:].split("/"):
        if not isinstance(node, dict) or part not in node:
            return schema
        node = node[part]
    return node if isinstance(node, dict) else schema


def validate(instance: Any, schema: dict[str, Any], *, root: dict[str, Any] | None = None, path: str = "") -> list[str]:
    """Return the problems with ``instance`` under ``schema``, as ``path: message`` strings (empty when valid)."""
    root = root if root is not None else schema
    schema = _resolve(schema, root)
    where = path or "value"
    errors: list[str] = []

    if "type" in schema:
        names = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_is_type(instance, n) for n in names):
            return [f"{where}: expected {' or '.join(names)}, got {_show(instance)}"]
    if "enum" in schema and instance not in schema["enum"]:
        return [f"{where}: must be one of {_show(schema['enum'])}, got {_show(instance)}"]
    if "const" in schema and instance != schema["const"]:
        return [f"{where}: must be {_show(schema['const'])}"]

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append(f"{where}: shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errors.append(f"{where}: longer than {schema['maxLength']} characters")
        if "pattern" in schema and not re.search(schema["pattern"], instance):
            errors.append(f"{where}: does not match pattern {schema['pattern']!r}")
    elif isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{where}: must be >= {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{where}: must be <= {schema['maximum']}")
        if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
            errors.append(f"{where}: must be > {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and instance >= schema["exclusiveMaximum"]:
            errors.append(f"{where}: must be < {schema['exclusiveMaximum']}")
    elif isinstance(instance, (list, tuple)):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(f"{where}: needs at least {schema['minItems']} items")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append(f"{where}: allows at most {schema['maxItems']} items")
        if isinstance(schema.get("items"), dict):
            for i, item in enumerate(instance):
                errors += validate(item, schema["items"], root=root, path=f"{where}[{i}]")
    elif isinstance(instance, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{where}.{key}: missing" if path else f"{key}: missing")
        for key, value in instance.items():
            sub = f"{where}.{key}" if path else key
            if key in props:
                errors += validate(value, props[key], root=root, path=sub)
            else:
                extra = schema.get("additionalProperties", True)
                if extra is False:
                    errors.append(f"{sub}: unexpected (known: {', '.join(props) or 'none'})")
                elif isinstance(extra, dict):
                    errors += validate(value, extra, root=root, path=sub)

    for key in ("anyOf", "oneOf"):
        if key in schema:
            matches = [not validate(instance, s, root=root, path=path) for s in schema[key]]
            if (key == "anyOf" and not any(matches)) or (key == "oneOf" and sum(matches) != 1):
                errors.append(f"{where}: does not match {key} ({'none' if not any(matches) else sum(matches)} matched)")
    for s in schema.get("allOf", []):
        errors += validate(instance, s, root=root, path=path)
    if "not" in schema and not validate(instance, schema["not"], root=root, path=path):
        errors.append(f"{where}: must not match the 'not' schema")
    return errors


def parse_json(text: str) -> tuple[Any, str | None]:
    """Parse a model's text as JSON. A ```json fence around it is tolerated. Returns (value, error)."""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[1] if "\n" in candidate else ""
        candidate = candidate.rsplit("```", 1)[0].strip()
    try:
        return json.loads(candidate), None
    except json.JSONDecodeError as exc:
        return None, f"not valid JSON ({exc.msg} at position {exc.pos})"
