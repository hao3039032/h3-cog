from h3_workflow import aligned_frames, dimensions


def test_ref2va_output_geometry_is_aligned():
    for aspect_ratio in ("16:9", "9:16", "1:1", "4:3", "3:4", "21:9"):
        assert all(value % 32 == 0 for value in dimensions(aspect_ratio, "preview"))
    for duration in (4, 5, 10, 15):
        assert aligned_frames(duration) % 17 == 5
