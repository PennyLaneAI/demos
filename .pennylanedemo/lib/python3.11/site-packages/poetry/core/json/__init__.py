from __future__ import annotations

import json
import sys

from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import fastjsonschema

from fastjsonschema.exceptions import JsonSchemaException


SCHEMA_DIR = Path(__file__).parent / "schemas"


if sys.version_info < (3, 9):

    def _get_schema_file(schema_name: str) -> Path:
        return SCHEMA_DIR / f"{schema_name}.json"

else:
    from importlib.resources import files

    if TYPE_CHECKING:
        from importlib.abc import Traversable

    def _get_schema_file(schema_name: str) -> Traversable:
        return files(__package__) / "schemas" / f"{schema_name}.json"


class ValidationError(ValueError):
    pass


def validate_object(obj: dict[str, Any], schema_name: str) -> list[str]:
    schema_file = _get_schema_file(schema_name)

    if not schema_file.is_file():
        raise ValueError(f"Schema {schema_name} does not exist.")

    with schema_file.open(encoding="utf-8") as f:
        schema = json.load(f)

    validate = fastjsonschema.compile(schema)

    errors = []
    try:
        validate(obj)
    except JsonSchemaException as e:
        errors = [e.message]

    return errors
