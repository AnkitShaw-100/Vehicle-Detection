from anpr.ocr import normalize_plate_text


def test_normalize_plate_text_basic():
    assert normalize_plate_text(" mh 12 ab 1234 ") == "MH12AB1234"


def test_normalize_plate_text_strips_non_alnum():
    assert normalize_plate_text("AB@12-34") == "AB1234"


def test_normalize_plate_text_empty():
    assert normalize_plate_text("") == ""
