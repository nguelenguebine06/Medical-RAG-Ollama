
import pandas as pd
from pathlib import Path

# Racine du projet
BASE_DIR = Path(__file__).resolve().parents[3]

# Chemin du dataset
DATA_PATH = BASE_DIR / "src" / "ml" / "intent_classifier" / "data" / "dataset_intentions.csv"


def load_dataset():

    df = pd.read_csv(DATA_PATH)

    required_columns = {"question", "intent"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Colonnes attendues : {required_columns}"
        )

    return df


def analyze_dataset(df):

    print("\n========== AUDIT DU DATASET ==========")

    # 1. Taille du dataset
    print("\n1. Dimensions")
    print(f"Nombre de lignes : {len(df)}")
    print(f"Nombre de colonnes : {len(df.columns)}")

    # 2. Valeurs manquantes
    print("\n2. Valeurs manquantes")
    print(df.isnull().sum())

    # 3. Questions vides
    empty_questions = (
        df["question"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    )

    print("\n3. Questions vides")
    print(empty_questions.sum())

    # 4. Distribution des intentions
    print("\n4. Distribution des intentions")
    distribution = df["intent"].value_counts()

    percentages = (
        df["intent"].value_counts(normalize=True) * 100
    )

    print(pd.DataFrame({
        "Nombre": distribution,
        "Pourcentage": percentages.round(2)
    }))

    # 5. Questions dupliquées
    print("\n5. Questions dupliquées")

    duplicates = df[
        df["question"].duplicated(keep=False)
    ]

    print(f"Nombre de lignes concernées : {len(duplicates)}")

    # 6. Questions identiques avec intentions différentes
    print("\n6. Incohérences d'annotations")

    conflicts = (
        df.groupby("question")["intent"]
        .nunique()
    )

    conflicts = conflicts[conflicts > 1]

    print(f"Questions avec plusieurs intentions : {len(conflicts)}")

    if not conflicts.empty:
        print(
            df[df["question"].isin(conflicts.index)]
            .sort_values("question")
            .to_string(index=False)
        )

    # 7. Longueur des questions
    print("\n7. Longueur des questions")

    lengths = df["question"].fillna("").astype(str).str.split().str.len()

    print(lengths.describe())

    print("\n========== FIN DE L'AUDIT ==========")

def inspect_samples(df, n=15):
    print("\n========== EXEMPLES PAR INTENTION ==========")

    for intent in df["intent"].unique():
        samples = df[df["intent"] == intent].sample(
            n=min(n, len(df[df["intent"] == intent])),
            random_state=42
        )

        print(f"\n--- INTENTION : {intent} ---")

        for index, row in samples.iterrows():
            print(f"- {row['question']}")

if __name__ == "__main__":
    dataset = load_dataset()
    analyze_dataset(dataset)
    inspect_samples(dataset)
