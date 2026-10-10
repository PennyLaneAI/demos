r"""*JSON schema to validate inventory dictionaries*.

This module is part of ``sphobjinv``,
a toolkit for manipulation and inspection of
Sphinx |objects.inv| files.

**Author**
    Brian Skinn (brian.skinn@gmail.com)

**File Created**
    7 Dec 2017

**Copyright**
    \(c) 2016-2026 Brian Skinn and community contributors

**Source Repository**
    https://github.com/bskinn/sphobjinv

**Documentation**
    https://sphobjinv.readthedocs.io/en/stable

**License**
    Code: `MIT License`_

    Docs & Docstrings: |CC BY 4.0|_

    See |license_txt|_ for full license terms.

**Members**

"""

# For jsonschema Draft 4.
# Schemas are defined with static field names as a versioning
# guarantee, instead of basing them dynamically on DataFields, etc.

# JSON dict schema
# Subschema for the inner data, both for clarity and to make it
# possible to satisfy flake8
subschema_json = {
    "name": {"type": "string"},
    "domain": {"type": "string"},
    "role": {"type": "string"},
    "priority": {"type": "string"},
    "uri": {"type": "string"},
    "dispname": {"type": "string"},
}

#: JSON schema for validating the |dict| forms of
#: Sphinx |objects.inv| inventories
#: as generated from or expected by
#: :class:`~sphobjinv.inventory.Inventory` classes.
json_schema = {
    "$schema": "https://json-schema.org/draft-04/schema#",
    "type": "object",
    "properties": {
        "project": {"type": "string"},
        "version": {"type": "string"},
        "count": {"type": "integer"},
        "metadata": {},
    },
    "patternProperties": {
        r"^\d+": {
            "type": "object",
            "properties": subschema_json,
            "additionalProperties": False,
            "required": list(subschema_json),
        }
    },
    "additionalProperties": False,
    "required": ["project", "version", "count"],
}
