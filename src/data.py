from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd
from pydantic import BaseModel
from sklearn.model_selection import GroupShuffleSplit

# Project root anchored to the repository directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEAN_DATA_DIR = DATA_DIR / "clean"

TRAIN_PATH = RAW_DATA_DIR / "AR_Hotels_Train_SB1.xml"
TEST_PATH = RAW_DATA_DIR / "AR_HOTE_SB1_TEST.xml.gold"

PATHS = {
    "train": TRAIN_PATH,
    "test": TEST_PATH,
}


class Opinion(BaseModel):
    target: str | None = None
    category: str
    polarity: str


def parse(paths: dict[str, Path | str] | Path | str | None = None) -> pd.DataFrame:
    if paths is None:
        target_paths = PATHS
    elif isinstance(paths, (str, Path)):
        target_paths = {"data": Path(paths)}
    else:
        target_paths = paths

    rows, sentence_id = [], 0
    for split_name, path in target_paths.items():
        for review in ET.parse(path).getroot().iter("Review"):
            for s in review.iter("sentence"):
                sentence_id += 1
                base = {
                    "split": split_name,
                    "review_id": int(review.get("rid")),
                    "sentence_id": sentence_id,
                    "text": s.findtext("text"),
                }
                opinions = [Opinion(**o.attrib).model_dump() for o in s.iter("Opinion")] or [{}]
                rows += [base | o for o in opinions]

    df = pd.DataFrame(rows)
    df[["entity", "attribute"]] = df.category.str.split("#", expand=True)

    if len(target_paths) == 1 and "data" in target_paths:
        df = df.drop(columns="split")

    return df


def load_train(path: Path | str | None = None) -> pd.DataFrame:
    return parse({"train": path or TRAIN_PATH})


def load_test(path: Path | str | None = None) -> pd.DataFrame:
    return parse({"test": path or TEST_PATH})


def load_data(paths: dict[str, Path | str] | None = None) -> pd.DataFrame:
    return parse(paths or PATHS)


def split(df: pd.DataFrame, val_size: float = 0.10, seed: int = 42):
    train_idx, val_idx = next(
        GroupShuffleSplit(test_size=val_size, random_state=seed).split(df, groups=df.review_id)
    )
    return df.iloc[train_idx], df.iloc[val_idx]


def sentences(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("sentence_id", sort=False)
        .agg(
            review_id=("review_id", "first"),
            text=("text", "first"),
            categories=("category", lambda c: sorted(c.dropna().unique())),
        )
        .reset_index()
    )
