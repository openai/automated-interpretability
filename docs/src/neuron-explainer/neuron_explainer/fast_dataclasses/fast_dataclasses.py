# Utilities for dataclasses that are very fast to serialize and deserialize, with limited data
# validation. Fields must not be tuples, since they get serialized and then deserialized as lists.
#
# The unit tests for this library show how to use it.

import json
from dataclasses import dataclass, field, fields, is_dataclass
from functools import partial
from typing import Any, Union

import orjson

dataclasses_by_name = {}
dataclasses_by_fieldnames = {}


@dataclass
class FastDataclass:
    dataclass_name: str = field(init=False)

    def __post_init__(self) -> None:
        self.dataclass_name = self.__class__.__name__


def register_dataclass(cls):  # type: ignore
    assert is_dataclass(cls), "Only dataclasses can be registered."
    dataclasses_by_name[cls.__name__] = cls
    name_set = frozenset(f.name for f in fields(cls) if f.name != "dataclass_name")
    dataclasses_by_fieldnames[name_set] = cls
    return cls


def dumps(obj: Any) -> bytes:
    return orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY)


def _object_hook(d: Any, backwards_compatible: bool = True) -> Any:
    # If d is a list, recurse.
    if isinstance(d, list):
        return [_object_hook(x, backwards_compatible=backwards_compatible) for x in d]
    # If d is not a dict, return it as is.
    if not isinstance(d, dict):
        return d
    cls = None
    if "dataclass_name" in d:
        if d["dataclass_name"] in dataclasses_by_name:
            cls = dataclasses_by_name[d["dataclass_name"]]
        else:
            assert backwards_compatible, (
                f"Dataclass {d['dataclass_name']} not found, set backwards_compatible=True if you "
                f"are okay with that."
            )
    # Load objects created without dataclass_name set.
    else:
        # Try our best to find a dataclass if backwards_compatible is True.
        if backwards_compatible:
            d_fields = frozenset(d.keys())
            if d_fields in dataclasses_by_fieldnames:
                cls = dataclasses_by_fieldnames[d_fields]
            elif len(d_fields) > 0:
                # Check if the fields are a subset of a dataclass (if the dataclass had extra fields
                # added since the data was created). Note that this will fail if fields were removed
                # from the dataclass.
                for key, possible_cls in dataclasses_by_fieldnames.items():
                    if d_fields.issubset(key):
                        cls = possible_cls
                        break
                else:
                    print(f"Could not find dataclass for {d_fields} {cls}")
    new_d = {
        k: _object_hook(v, backwards_compatible=backwards_compatible)
        for k, v in d.items()
        if k != "dataclass_name"
    }
    if cls is not None:
        return cls(**new_d)
    else:
        return new_d


def loads(s: Union[str, bytes], backwards_compatible: bool = True) -> Any:
    return json.loads(
        s,
        object_hook=partial(_object_hook, backwards_compatible=backwards_compatible),
    )
