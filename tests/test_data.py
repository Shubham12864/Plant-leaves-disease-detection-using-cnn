import pandas as pd
import pytest
from PIL import Image

from plant_disease.data import PlantVillageImages, load_split, resolve_image_root


def _make_repo(tmp_path, train_files, test_files):
    base = tmp_path / "data" / "raw" / "PlantVillage"
    for sub, files in (("train", train_files), ("test", test_files)):
        for rel in files:
            path = base / sub / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), (10, 200, 30)).save(path)
    return tmp_path


def test_resolve_image_root_picks_train_for_development_rows(tmp_path):
    root = _make_repo(tmp_path, ["a/1.jpg"], ["b/2.jpg"])
    frame = pd.DataFrame({"relative_path": ["a/1.jpg"]})
    assert resolve_image_root(root, frame).name == "train"


def test_resolve_image_root_picks_test_for_published_test_rows(tmp_path):
    root = _make_repo(tmp_path, ["a/1.jpg"], ["b/2.jpg"])
    frame = pd.DataFrame({"relative_path": ["b/2.jpg"]})
    assert resolve_image_root(root, frame).name == "test"


def test_resolve_image_root_raises_when_nothing_matches(tmp_path):
    root = _make_repo(tmp_path, ["a/1.jpg"], [])
    frame = pd.DataFrame({"relative_path": ["missing.jpg"]})
    with pytest.raises(FileNotFoundError):
        resolve_image_root(root, frame)


def test_resolve_image_root_rejects_empty_manifest(tmp_path):
    root = _make_repo(tmp_path, ["a/1.jpg"], [])
    with pytest.raises(ValueError):
        resolve_image_root(root, pd.DataFrame({"relative_path": []}))


def test_load_split_refuses_the_locked_test_split(tmp_path):
    with pytest.raises(ValueError):
        load_split(tmp_path, "test")


def test_load_split_returns_only_requested_split_with_sorted_classes(tmp_path):
    manifests = tmp_path / "data" / "manifests"
    manifests.mkdir(parents=True)
    pd.DataFrame(
        {
            "split": ["train", "train", "validation"],
            "relative_path": ["b/1.jpg", "a/2.jpg", "a/3.jpg"],
            "label": ["b", "a", "a"],
        }
    ).to_csv(manifests / "plantvillage_development.csv", index=False)
    frame, classes = load_split(tmp_path, "train")
    assert len(frame) == 2
    assert classes == ["a", "b"]


def test_dataset_yields_normalised_tensor_and_label_index(tmp_path):
    root = _make_repo(tmp_path, ["a/1.jpg"], [])
    frame = pd.DataFrame({"relative_path": ["a/1.jpg"], "label": ["a"]})
    dataset = PlantVillageImages(root, frame, ["a", "b"], "resnet18", image_size=16)
    tensor, label = dataset[0]
    assert tuple(tensor.shape) == (3, 16, 16)
    assert label == 0
    assert len(dataset) == 1
