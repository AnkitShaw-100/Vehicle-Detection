from anpr.video_runner import pad_and_clip_bbox


def test_pad_and_clip_bbox_clamps_to_frame():
    x1, y1, x2, y2 = pad_and_clip_bbox((10, 10, 20, 20), pad=0.5, w=25, h=25)
    assert 0 <= x1 < x2 <= 25
    assert 0 <= y1 < y2 <= 25


def test_pad_and_clip_bbox_handles_negative_and_overflow():
    x1, y1, x2, y2 = pad_and_clip_bbox((-10, -10, 40, 40), pad=0.0, w=30, h=30)
    assert (x1, y1, x2, y2) == (0, 0, 30, 30)
