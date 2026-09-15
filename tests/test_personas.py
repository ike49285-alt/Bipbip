import pytest

from thirsttrap import personas


def test_default_persona_exists():
    assert personas.DEFAULT_PERSONA in personas.PERSONAS


def test_names_are_sorted_and_complete():
    assert personas.names() == sorted(personas.PERSONAS)


def test_lookup_is_case_and_whitespace_insensitive():
    assert personas.get("  FLIRT ") is personas.PERSONAS["flirt"]


def test_unknown_persona_lists_the_valid_ones():
    with pytest.raises(KeyError) as excinfo:
        personas.get("smoulder")
    message = excinfo.value.args[0]
    assert "smoulder" in message
    for name in personas.names():
        assert name in message


@pytest.mark.parametrize("name", personas.names())
def test_every_persona_is_usably_specified(name):
    persona = personas.PERSONAS[name]
    assert persona.name == name
    assert persona.summary
    # A one-line directive gives the model nothing to work with.
    assert len(persona.directive) > 60
