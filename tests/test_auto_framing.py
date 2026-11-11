from dataclasses import replace

import numpy as np
import pytest

from framepilot.calibration import Calibration
from framepilot.framing import WHOLE, AutoFraming, FramingRule, freeze_framing
from framepilot.geometry import Rect
from framepilot.pipeline import Pipeline, Renderer, SceneSlot, Settings

PART = Rect(0, 0, 0.5, 1)


def image_and_subject():
    image = np.zeros((180, 320, 3), np.uint8)
    image[50:130, 80:140] = (0, 0, 240)
    image[50:130, 140:200] = (240, 0, 0)
    return image, Rect(0.25, 50 / 180, 120 / 320, 80 / 180)


@pytest.mark.parametrize("direction", ["larger", "smaller"])
def test_area_hysteresis_prevents_rapid_part_whole_switching(direction):
    engine = AutoFraming()
    rule = FramingRule(True, direction, threshold=0.2, hysteresis=0.1, transition=0)
    starting_area, other_area = (0.1, 0.3) if direction == "larger" else (0.3, 0.1)
    assert engine.update(PART, starting_area, rule, 0) == PART
    for index in range(1, 50):
        assert engine.update(PART, 0.2 + 0.005 * np.sin(index), rule, index / 30) == PART
    assert engine.update(PART, other_area, rule, 2) == WHOLE
    for index in range(1, 50):
        assert engine.update(PART, 0.2 + 0.005 * np.sin(index), rule, 2 + index / 30) == WHOLE
    assert engine.update(PART, starting_area, rule, 4) == PART


def test_transition_and_reversal_are_continuous_and_time_seek_resets_state():
    rule = FramingRule(enabled=True, transition=0.2)
    engine = AutoFraming()
    assert engine.update(PART, 0.1, rule, 0) == PART
    assert engine.update(PART, 0.4, rule, 1) == PART
    halfway = engine.update(PART, 0.4, rule, 1.1)
    assert halfway.w == pytest.approx(0.75)
    assert engine.active
    reversed_start = engine.update(PART, 0.1, rule, 1.1)
    assert reversed_start == halfway
    assert engine.update(PART, 0.1, rule, 1.31) == PART
    assert not engine.active
    assert engine.update(PART, 0.4, rule, 0) == WHOLE


def test_remembering_current_frame_as_threshold_starts_with_part_in_the_deadband():
    engine = AutoFraming()
    rule = FramingRule(enabled=True, threshold=0.2)
    assert engine.update(PART, 0.2, rule, 0) == PART
    engine.update(PART, 0.23, rule, 1)
    assert engine.whole


@pytest.mark.parametrize("direction", ["larger", "smaller"])
def test_zero_hysteresis_does_not_toggle_at_exactly_the_threshold(direction):
    engine = AutoFraming()
    rule = FramingRule(enabled=True, whole_when=direction, hysteresis=0, transition=0)
    for index in range(20):
        assert engine.update(PART, rule.threshold, rule, index / 30) == WHOLE


def test_transition_does_not_jump_at_start_with_camera_fill_and_calibration():
    image, subject = image_and_subject()
    settings = Settings(
        width=400,
        height=240,
        target=Rect(0, 0, 0.2, 0.5),
        subject_region=PART,
        framing=FramingRule(enabled=True),
        fill=True,
        calibration=Calibration(zoom=1.5, offset_x=0.2, offset_y=0.1),
    )
    renderer = Renderer()
    before = renderer.render(image, subject, settings, 0)
    large = replace(settings, target=Rect(0, 0, 0.8, 0.8))
    start = renderer.render(image, subject, large, 0.1)
    np.testing.assert_array_equal(start, before)
    assert renderer.last_framing["transitioning"]
    after = renderer.render(image, subject, large, 0.3)
    assert after[:, :, 0].max() > 230 and after[:, :, 2].max() > 230


@pytest.mark.parametrize("shape", ["rectangle", "ellipse"])
def test_whole_view_is_not_cropped_by_fill_zoom_offsets_or_motion(shape):
    image, subject = image_and_subject()
    rule = FramingRule(enabled=True, threshold=0.1, transition=0)
    settings = Settings(
        width=400,
        height=240,
        target=Rect(0.1, 0.1, 0.8, 0.8),
        subject_mode="object",
        subject_region=PART,
        shape=shape,
        fill=True,
        padding=0,
        framing=rule,
        calibration=Calibration(zoom=2, offset_x=0.4, offset_y=0.3),
    )
    renderer = Renderer()
    output = renderer.render(image, subject, settings, 0)
    assert renderer.last_framing["whole"]
    assert output[:, :, 0].max() > 230 and output[:, :, 2].max() > 230
    # Every corner of the whole target must lie within the rectangle / inscribed ellipse.
    for moved in (subject, Rect(subject.x + 0.2, subject.y, subject.w, subject.h)):
        renderer.render(image, moved, settings, 1 / 30)
        placement = renderer.placement
        bounds = settings.calibration.crop.pixels(400, 240)
        for x in (moved.x * 320, (moved.x + moved.w) * 320):
            for y in (moved.y * 180, (moved.y + moved.h) * 180):
                px, py = x * placement.scale + placement.x, y * placement.scale + placement.y
                assert bounds.x - 1e-6 <= px <= bounds.x + bounds.w + 1e-6
                assert bounds.y - 1e-6 <= py <= bounds.y + bounds.h + 1e-6
                if shape == "ellipse":
                    assert ((px - bounds.center[0]) / (bounds.w / 2)) ** 2 + (
                        (py - bounds.center[1]) / (bounds.h / 2)
                    ) ** 2 <= 1


def test_camera_mapping_reacts_to_receiving_frame_area_and_export_keeps_hysteresis_state():
    image, subject = image_and_subject()
    settings = Settings(
        width=400,
        height=240,
        subject_mode="object",
        padding=0,
        subject_region=PART,
        framing=FramingRule(True, transition=0),
        target=Rect(0, 0, 0.2, 0.5),
    )
    pipeline = Pipeline()
    output, found, _ = pipeline.process(image, 0, settings, subject, True)
    assert output[:, :, 0].max() == 0
    assert found == subject
    whole_settings = replace(settings, target=Rect(0, 0, 0.8, 0.8))
    output, found, _ = pipeline.process(image, 0.1, whole_settings, None, False, repaint=True)
    assert found == subject
    assert pipeline.last_framing["whole"]
    assert output[:, :, 0].max() > 230
    # Area 19% retains whole after a large frame, although a fresh decision at 19% would choose part.
    band = replace(settings, target=Rect(0, 0, 0.38, 0.5))
    pipeline.process(image, 0.2, band, None, False, repaint=True)
    assert pipeline.last_framing["whole"]
    frozen = freeze_framing(band, pipeline.last_tracks)
    assert frozen.subject_region == WHOLE and frozen.whole_fit
    assert not frozen.framing.enabled
    exported_frame = Renderer().render(image, subject, frozen, 0)
    assert exported_frame[:, :, 0].max() > 230
    assert pipeline.object_tracker.box == subject


def test_each_slot_has_an_independent_rule_without_losing_target():
    image, subject = image_and_subject()
    slots = (
        SceneSlot(
            "a",
            Rect(0, 0, 0.4, 0.8),
            subject_mode="object",
            subject_region=PART,
            framing=FramingRule(True, "larger", threshold=0.2, transition=0),
        ),
        SceneSlot(
            "b",
            Rect(0.5, 0, 0.4, 0.8),
            subject_mode="object",
            subject_region=PART,
            framing=FramingRule(True, "smaller", threshold=0.2, transition=0),
        ),
    )
    settings = Settings(width=400, height=240, slots=slots)
    pipeline = Pipeline()
    pipeline.process(image, 0, settings, {"a": subject, "b": subject}, True)
    assert pipeline.last_tracks["a"]["framing"]["whole"]
    assert not pipeline.last_tracks["b"]["framing"]["whole"]
    changed = replace(
        settings, slots=tuple(replace(slot, target=Rect(slot.target.x, 0, 0.2, 0.5)) for slot in slots)
    )
    pipeline.process(image, 0.1, changed, None, False, repaint=True)
    assert not pipeline.last_tracks["a"]["framing"]["whole"]
    assert pipeline.last_tracks["b"]["framing"]["whole"]
    assert all(info["subject"] == subject for info in pipeline.last_tracks.values())


def test_paused_media_uses_presentation_time_to_complete_auto_transition():
    image, subject = image_and_subject()
    settings = Settings(
        width=400,
        height=240,
        subject_mode="object",
        subject_region=PART,
        framing=FramingRule(enabled=True, transition=0.12),
        target=Rect(0, 0, 0.2, 0.5),
    )
    pipeline = Pipeline()
    pipeline.process(image, 0, settings, subject, True, render_time=10)
    settings = replace(settings, target=Rect(0, 0, 0.8, 0.8))
    pipeline.process(image, 0, settings, None, False, repaint=True, render_time=10.1)
    assert pipeline.last_framing["transitioning"]
    pipeline.process(image, 0, settings, None, False, repaint=True, render_time=10.3)
    assert not pipeline.last_framing["transitioning"]
    assert pipeline.last_framing["region"] == [0, 0, 1, 1]
    assert pipeline.object_tracker.box == subject
