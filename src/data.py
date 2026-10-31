from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEAN_DATA_DIR = DATA_DIR / "clean"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_PATHS = {
    "train": RAW_DATA_DIR / "AR_Hotels_Train_SB1.xml",
    "test": RAW_DATA_DIR / "AR_HOTE_SB1_TEST.xml.gold",
}
CLEAN_TRAIN_PATH = CLEAN_DATA_DIR / "hotels_absa_train.csv"
CLEAN_TEST_PATH = CLEAN_DATA_DIR / "hotels_absa_test.csv"

POLARITIES = ["negative", "neutral", "positive"]


class Opinion(BaseModel):
    target: str | None = None
    category: str
    polarity: str


def parse(paths: dict[str, Path] = RAW_PATHS) -> pd.DataFrame:
    """One row per opinion. A sentence without opinion keeps one row with empty labels."""
    rows, sentence_id = [], 0
    for split_name, path in paths.items():
        for review in ET.parse(path).getroot().iter("Review"):
            for sentence in review.iter("sentence"):
                sentence_id += 1
                base = {
                    "split": split_name,
                    "review_id": int(review.get("rid")),
                    "sentence_id": sentence_id,
                    "text": sentence.findtext("text"),
                }
                opinions = [Opinion(**o.attrib).model_dump() for o in sentence.iter("Opinion")] or [{}]
                rows += [base | opinion for opinion in opinions]

    df = pd.DataFrame(rows)
    df[["entity", "attribute"]] = df.category.str.split("#", expand=True)
    return df


def load_opinions(path: Path) -> pd.DataFrame:
    # keep_default_na=False: otherwise the target "NULL" (implicit aspect) is read as missing
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def split(df: pd.DataFrame, val_size: float = 0.10, seed: int = 42):
    splitter = GroupShuffleSplit(test_size=val_size, random_state=seed)
    train_index, val_index = next(splitter.split(df, groups=df.review_id))
    return df.iloc[train_index], df.iloc[val_index]


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


def load_train_val(seed: int = 42):
    """Clean train file -> (all opinions, train sentences, val sentences), split by review."""
    opinions = load_opinions(CLEAN_TRAIN_PATH)
    train, val = split(sentences(opinions), seed=seed)
    return opinions, train, val


def category_list(df: pd.DataFrame) -> list[str]:
    return sorted(df.category.dropna().unique())


def category_matrix(category_lists, classes: list[str]) -> np.ndarray:
    """Lists of category names -> 0/1 matrix (n_sentences, n_classes)."""
    index = {name: i for i, name in enumerate(classes)}
    matrix = np.zeros((len(category_lists), len(classes)), dtype=int)
    for row, categories in enumerate(category_lists):
        for category in categories:
            matrix[row, index[category]] = 1
    return matrix


def pairs(df: pd.DataFrame, sentence_table: pd.DataFrame) -> pd.DataFrame:
    """One row per opinion of the sentences in sentence_table."""
    columns = ["sentence_id", "review_id", "text", "target", "category", "polarity"]
    table = df[columns].dropna(subset=["category", "polarity"])
    table = table[table.sentence_id.isin(sentence_table.sentence_id)]
    return table.reset_index(drop=True)


def pair_units(pair_table: pd.DataFrame) -> pd.DataFrame:
    """One row per (sentence, category). Opinions that disagree on the polarity are marked ambiguous."""
    rows = []
    for (sentence_id, category), group in pair_table.groupby(["sentence_id", "category"], sort=False):
        polarities = sorted(group.polarity.unique())
        rows.append({
            "sentence_id": sentence_id,
            "review_id": group.review_id.iloc[0],
            "text": group.text.iloc[0],
            "category": category,
            "ambiguous": len(polarities) > 1,
            "polarity": polarities[0] if len(polarities) == 1 else None,
        })
    return pd.DataFrame(rows)
