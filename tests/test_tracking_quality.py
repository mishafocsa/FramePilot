import numpy as np

from framepilot.geometry import iou
from framepilot.objects import ObjectTracker
from framepilot.tracking_validation import rigid_frame, validate_tracking


def test_planar_rotation_keeps_original_reference_and_conservative_bounds():
    tracker = ObjectTracker()
    reference = None
    for index in range(90):
        image, expected = rigid_frame(index, angle=index)
        found, _ = tracker.update(image, index / 30, expected if index == 0 else None)
        assert found is not None, (index, tracker.diagnostics())
        assert iou(found, expected) > 0.85, (index, found, expected)
        if reference is None:
            reference = tracker.verification.copy()
        np.testing.assert_array_equal(tracker.verification, reference)
    assert tracker.metrics["rotation_degrees"] > 85


def test_lookalikes_after_occlusion_are_never_automatically_chosen_and_click_resolves():
    tracker = ObjectTracker()
    image, selected = rigid_frame()
    tracker.update(image, 0, selected)
    tracker.update(np.full_like(image, 22), 0.1)
    image, _ = rigid_frame(twin=True)
    assert tracker.update(image, 0.2)[0] is None
    assert tracker.metrics["state"] == "ambiguous"
    assert len(tracker.proposals) >= 2
    for index in range(3, 8):
        assert tracker.update(image, index / 10)[0] is None
    chosen = max(tracker.proposals, key=lambda rect: rect.x)
    assert iou(tracker.update(image, 0.9, chosen)[0], chosen) > 0.95
    assert tracker.update(image, 1.0)[0] is not None


def test_duplicate_timestamp_cannot_confirm_recovery():
    tracker = ObjectTracker()
    image, selected = rigid_frame()
    tracker.update(image, 0, selected)
    tracker.update(np.full_like(image, 22), 0.1)
    assert tracker.update(image, 0.2)[0] is None
    assert tracker.metrics["state"] == "recovering"
    assert tracker.update(image, 0.2)[0] is None
    assert tracker.update(image, 0.24)[0] is not None


def test_pure_tracker_acceptance_cases_have_ground_truth_and_no_false_reacquisition():
    result = validate_tracking()
    assert result["passed"], result


def test_repaint_does_not_inflate_object_quality_or_advance_confirmation():
    from framepilot.pipeline import Pipeline, Settings

    pipeline = Pipeline()
    image, selected = rigid_frame()
    settings = Settings(subject_mode="object", width=320, height=180)
    pipeline.process(image, 0, settings, selected, True)
    before = dict(pipeline.last_quality)
    pipeline.process(image, 0, settings, None, False, repaint=True)
    assert pipeline.last_quality == before
    assert pipeline.last_tracks["single"]["tracking"] == before
