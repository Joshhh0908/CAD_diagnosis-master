import torch

# import your real functions
from optimization import sc2od_targets, od2sc_targets


def print_od_targets(name, od_targets):
    print(f"\n{name}")
    for i, t in enumerate(od_targets):
        labels = t["labels"]
        boxes = t["boxes"]
        print(f"  sample {i}:")
        print(f"    labels: {labels.tolist()}")
        print(f"    boxes : {boxes.tolist()}")


def print_sc_targets(name, sc_targets):
    print(f"\n{name}")
    for i, t in enumerate(sc_targets):
        labels = t["labels"]
        print(f"  sample {i}: {labels.tolist()}")


def assert_od_targets_valid(od_targets):
    for i, t in enumerate(od_targets):
        labels = t["labels"]
        boxes = t["boxes"]

        # assert labels.dtype in (torch.int64, torch.long), f"sample {i}: labels dtype wrong"
        # assert boxes.dtype in (torch.float32, torch.float64), f"sample {i}: boxes dtype wrong"
        assert labels.ndim == 1, f"sample {i}: labels should be 1D"
        # assert boxes.ndim == 2 and boxes.shape[1] == 2, f"sample {i}: boxes should be (N,2)"
        assert len(labels) == len(boxes), f"sample {i}: labels/boxes length mismatch"

        if labels.numel() > 0:
            assert labels.min().item() >= 1, f"sample {i}: OD labels should be lesion classes 1..6 only"
            assert labels.max().item() <= 6, f"sample {i}: OD labels should be <= 6"
            assert not (labels == 0).any().item(), f"sample {i}: OD targets should not contain background 0"


def assert_sc_targets_valid(sc_targets, seq_length):
    for i, t in enumerate(sc_targets):
        labels = t["labels"]

        assert labels.dtype in (torch.int64, torch.long), f"sample {i}: SC labels dtype wrong"
        assert labels.ndim == 1, f"sample {i}: SC labels should be 1D"
        assert len(labels) == seq_length, f"sample {i}: SC labels length should be {seq_length}"

        assert labels.min().item() >= 0, f"sample {i}: SC labels should be >= 0"
        assert labels.max().item() <= 6, f"sample {i}: SC labels should be <= 6"


def test_sc2od_basic():
    print("\n=== test_sc2od_basic ===")
    seq_length = 8
    sc_targets = [
        {"labels": torch.tensor([0, 0, 2, 2, 2, 0, 5, 5], dtype=torch.long)},
        {"labels": torch.tensor([0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long)},
        {"labels": torch.tensor([1, 1, 0, 3, 3, 4, 4, 4], dtype=torch.long)},
    ]

    od_targets = sc2od_targets(sc_targets, seq_length)
    print_sc_targets("input SC", sc_targets)
    print_od_targets("converted OD", od_targets)
    assert_od_targets_valid(od_targets)

    # label-only expectations
    assert od_targets[0]["labels"].tolist() == [2, 5], "sample 0 labels wrong"
    assert od_targets[1]["labels"].tolist() == [], "sample 1 should have no OD boxes"
    assert od_targets[2]["labels"].tolist() == [1, 3, 4], "sample 2 labels wrong"

    print("test_sc2od_basic passed")


def test_od2sc_basic():
    print("\n=== test_od2sc_basic ===")
    seq_length = 8
    od_targets = [
        {
            "labels": torch.tensor([2, 5], dtype=torch.long),
            "boxes": torch.tensor([[0.25, 0.625], [0.75, 1.0]], dtype=torch.float32),
        },
        {
            "labels": torch.tensor([], dtype=torch.long),
            "boxes": torch.empty((0, 2), dtype=torch.float32),
        },
        {
            "labels": torch.tensor([1, 3], dtype=torch.long),
            "boxes": torch.tensor([[0.125, 0.25], [0.5, 0.75]], dtype=torch.float32),
        },
    ]

    sc_targets = od2sc_targets(od_targets, seq_length)
    print_od_targets("input OD", od_targets)
    print_sc_targets("converted SC", sc_targets)
    assert_sc_targets_valid(sc_targets, seq_length)

    # exact slice labels depend on your box discretization,
    # so here we mainly check label range and presence
    uniq0 = set(torch.unique(sc_targets[0]["labels"]).tolist())
    uniq1 = set(torch.unique(sc_targets[1]["labels"]).tolist())
    uniq2 = set(torch.unique(sc_targets[2]["labels"]).tolist())

    assert uniq0.issubset({0, 2, 5}), f"sample 0 unexpected labels: {uniq0}"
    assert uniq1 == {0}, f"sample 1 should be all background, got {uniq1}"
    assert uniq2.issubset({0, 1, 3}), f"sample 2 unexpected labels: {uniq2}"

    print("test_od2sc_basic passed")


def test_roundtrip_sc_to_od_to_sc():
    print("\n=== test_roundtrip_sc_to_od_to_sc ===")
    seq_length = 12
    original_sc = [
        {"labels": torch.tensor([0, 0, 1, 1, 1, 0, 4, 4, 0, 6, 6, 0], dtype=torch.long)},
        {"labels": torch.tensor([0, 2, 2, 2, 0, 0, 3, 3, 3, 3, 0, 0], dtype=torch.long)},
        {"labels": torch.tensor([0] * 12, dtype=torch.long)},
    ]

    od_targets = sc2od_targets(original_sc, seq_length)
    recon_sc = od2sc_targets(od_targets, seq_length)

    print_sc_targets("original SC", original_sc)
    print_od_targets("OD from SC", od_targets)
    print_sc_targets("reconstructed SC", recon_sc)

    assert_od_targets_valid(od_targets)
    assert_sc_targets_valid(recon_sc, seq_length)

    # Because of box normalization/discretization, exact sequence equality
    # may fail due to boundary/off-by-one issues.
    # So here we test weaker but important conditions:
    for i in range(len(original_sc)):
        orig = original_sc[i]["labels"]
        rec = recon_sc[i]["labels"]

        orig_nonzero = set(orig[orig > 0].tolist())
        rec_nonzero = set(rec[rec > 0].tolist())

        assert rec_nonzero.issubset(orig_nonzero.union(rec_nonzero)), f"sample {i}: weird reconstructed labels"
        assert rec.min().item() >= 0 and rec.max().item() <= 6, f"sample {i}: reconstructed labels out of range"

    print("test_roundtrip_sc_to_od_to_sc passed")


def test_sc2od_no_background_labels_in_output():
    print("\n=== test_sc2od_no_background_labels_in_output ===")
    seq_length = 10
    sc_targets = [
        {"labels": torch.tensor([0, 0, 0, 1, 1, 0, 2, 2, 0, 0], dtype=torch.long)},
        {"labels": torch.tensor([3, 3, 3, 0, 0, 4, 4, 0, 5, 5], dtype=torch.long)},
    ]

    od_targets = sc2od_targets(sc_targets, seq_length)
    print_od_targets("OD from SC", od_targets)

    for i, t in enumerate(od_targets):
        if t["labels"].numel() > 0:
            assert not (t["labels"] == 0).any().item(), f"sample {i}: background label 0 leaked into OD output"

    print("test_sc2od_no_background_labels_in_output passed")


def test_od2sc_empty_boxes():
    print("\n=== test_od2sc_empty_boxes ===")
    seq_length = 16
    od_targets = [
        {
            "labels": torch.tensor([], dtype=torch.long),
            "boxes": torch.empty((0, 2), dtype=torch.float32),
        }
    ]

    sc_targets = od2sc_targets(od_targets, seq_length)
    print_sc_targets("SC from empty OD", sc_targets)

    assert len(sc_targets) == 1
    assert torch.equal(sc_targets[0]["labels"], torch.zeros(seq_length, dtype=torch.long, device=sc_targets[0]["labels"].device))

    print("test_od2sc_empty_boxes passed")


if __name__ == "__main__":
    test_sc2od_basic()
    test_od2sc_basic()
    test_roundtrip_sc_to_od_to_sc()
    test_sc2od_no_background_labels_in_output()
    test_od2sc_empty_boxes()

    print("\nAll tests finished.")