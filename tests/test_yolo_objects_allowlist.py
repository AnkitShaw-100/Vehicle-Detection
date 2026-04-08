from anpr.yolo_objects import build_class_allowlist


def test_coco_subset_human_car_truck():
    # Minimal fake COCO-like names
    names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 7: "truck"}
    allowed, id_disp, warns = build_class_allowlist(names, ["human", "car", "truck"])
    assert allowed == {0, 2, 7}
    assert id_disp[0] == "Human"
    assert id_disp[2] == "Car"
    assert id_disp[7] == "Truck"
    assert not any("ambulance" in w.lower() for w in warns)


def test_vehicle_keyword_groups_labels():
    names = {0: "person", 2: "car", 3: "motorcycle", 7: "truck", 5: "bus"}
    allowed, id_disp, _ = build_class_allowlist(names, ["human", "vehicle"])
    assert 0 in allowed
    assert allowed >= {2, 3, 5, 7}
    assert id_disp[0] == "Human"
    for cid in (2, 3, 5, 7):
        assert id_disp[cid] == "Vehicle"


def test_ambulance_missing_warns():
    names = {0: "person", 2: "car"}
    allowed, _, warns = build_class_allowlist(names, ["ambulance", "car"])
    assert 2 in allowed
    assert any("ambulance" in w.lower() for w in warns)
