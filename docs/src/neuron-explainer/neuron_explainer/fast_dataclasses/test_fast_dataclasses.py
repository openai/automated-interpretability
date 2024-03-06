from dataclasses import dataclass

import pytest

from .fast_dataclasses import FastDataclass, dumps, loads, register_dataclass


# Inheritance is a bit tricky with our setup. dataclass_name must be set for instances of these
# classes to serialize and deserialize correctly, but if it's given a default value, then subclasses
# can't have any fields that don't have default values, because of how constructors are generated
# for dataclasses (fields with no default value can't follow those with default values). To work
# around this, we set dataclass_name in __post_init__ on the base class, which is called after the
# constructor. The implementation does the right thing for both the base class and the subclass.
@register_dataclass
@dataclass
class DataclassC(FastDataclass):
    ints: list[int]


@register_dataclass
@dataclass
class DataclassC_ext(DataclassC):
    s: str


@register_dataclass
@dataclass
class DataclassB(FastDataclass):
    str_to_c: dict[str, DataclassC]
    cs: list[DataclassC]


@register_dataclass
@dataclass
class DataclassA(FastDataclass):
    floats: list[float]
    strings: list[str]
    bs: list[DataclassB]


@register_dataclass
@dataclass
class DataclassD(FastDataclass):
    s1: str
    s2: str = "default"


def test_dataclasses() -> None:
    a = DataclassA(
        floats=[1.0, 2.0],
        strings=["a", "b"],
        bs=[
            DataclassB(
                str_to_c={"a": DataclassC(ints=[1, 2]), "b": DataclassC(ints=[3, 4])},
                cs=[DataclassC(ints=[5, 6]), DataclassC_ext(ints=[7, 8], s="s")],
            ),
            DataclassB(
                str_to_c={"c": DataclassC_ext(ints=[9, 10], s="t"), "d": DataclassC(ints=[11, 12])},
                cs=[DataclassC(ints=[13, 14]), DataclassC(ints=[15, 16])],
            ),
        ],
    )
    assert loads(dumps(a)) == a


def test_c_and_c_ext() -> None:
    c_ext = DataclassC_ext(ints=[3, 4], s="s")
    assert loads(dumps(c_ext)) == c_ext

    c = DataclassC(ints=[1, 2])
    assert loads(dumps(c)) == c


def test_bad_serialized_data() -> None:
    assert type(loads(dumps(DataclassC(ints=[3, 4])))) == DataclassC
    assert type(loads('{"ints": [3, 4]}', backwards_compatible=False)) == dict
    assert type(loads('{"ints": [3, 4], "dataclass_name": "DataclassC"}')) == DataclassC
    with pytest.raises(TypeError):
        loads('{"ints": [3, 4], "bogus_extra_field": "foo", "dataclass_name": "DataclassC"}')
    with pytest.raises(TypeError):
        loads('{"ints_field_is_missing": [3, 4], "dataclass_name": "DataclassC"}')
    assert type(loads('{"s1": "test"}', backwards_compatible=False)) == dict
    assert type(loads('{"s1": "test"}', backwards_compatible=True)) == DataclassD
