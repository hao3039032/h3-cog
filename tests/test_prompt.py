import pytest

from h3_prompt import format_h3_prompt
from h3_workflow import aligned_frames, dimensions, validate_inputs


def test_duration_grid_and_native_dimensions():
    assert aligned_frames(5) == 124
    assert aligned_frames(15) == 362
    assert dimensions("16:9", "native") == (1344, 768)
    assert dimensions("9:16", "native") == (768, 1344)
    with pytest.raises(ValueError, match="between 4 and 15"):
        aligned_frames(3)


def test_prompt_modes_follow_official_keyframe_prefixes():
    plain = format_h3_prompt("A fox runs through snow.", 5.17)
    assert plain.startswith("integrated_multimodal_description: [Shot 1]")
    first = format_h3_prompt("The fox turns.", 5.17, first_frame=True)
    assert "at 0.00 seconds" in first and "<Picture 1>" in first
    both = format_h3_prompt("The fox returns.", 5.17, first_frame=True, last_frame=True)
    assert "Picture 1" in both and "Picture 2" in both and "5.17-second" in both


def test_structured_prompt_is_not_double_wrapped():
    source = "integrated_multimodal_description: [Shot 1] test\n\noverall_soundscape: N/A\n\nnon_diegetic_music: N/A"
    assert format_h3_prompt(source, 5.17, first_frame=True) == source


def test_loop_is_keyframe_conditioned():
    with pytest.raises(ValueError, match="requires first_frame"):
        validate_inputs(first_frame=None, last_frame=None, loop=True, steps=20, seed=1)
    with pytest.raises(ValueError, match="cannot be combined"):
        validate_inputs(first_frame="first.png", last_frame="last.png", loop=True, steps=20, seed=1)
