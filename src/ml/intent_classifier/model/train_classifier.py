import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, ConfusionMatrixDisplay
)

DATA_PATH  = "ml/intent_classifier/data/dataset_intentions.csv"
MODEL_PATH = "ml/intent_classifier/model/intent_classifier.pkl"

def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset chargé : {len(df)} exemples")
    print(f"Distribution des intentions :")
    print(df['intent'].value_counts())
    print()
    return df

def add_features(df):
    import re
    CODE_RE = re.compile(r'\b[A-Z]{1,3}[-]?\d{1,4}\b', re.IGNORECASE)

    df = df.copy()
    df['has_code']       = df['question'].apply(lambda q: int(bool(CODE_RE.search(q))))
    df['has_question']   = df['question'].apply(lambda q: int('?' in q))
    df['word_count']     = df['question'].apply(lambda q: len(q.split()))
    df['has_verb_inter'] = df['question'].apply(
        lambda q: int(any(w in q.lower() for w in ['quel', 'comment', 'pourquoi', 'quels', 'quelle']))
    )
    return df

def build_pipeline(model_type='logistic'):
    tfidf = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 4),
        max_features=500,
        sublinear_tf=True
    )

    if model_type == 'logistic':
        clf = LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=1.0
        )
    else:
        from sklearn.calibration import CalibratedClassifierCV
        clf = CalibratedClassifierCV(
        SVC(kernel='rbf', random_state=42),
        ensemble=False
        )

    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('clf',   clf)
    ])
    return pipeline

def train_and_evaluate(df):
    X = df['question']
    y = df['intent']

    # Split train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train : {len(X_train)} exemples")
    print(f"Test  : {len(X_test)} exemples\n")

    results = {}

    for model_name in ['logistic', 'svm']:
        print(f"--- Modèle : {model_name} ---")
        pipeline = build_pipeline(model_name)

        # Validation croisée sur le train
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
        print(f"CV accuracy (5-fold) : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        # Entraînement final
        pipeline.fit(X_train, y_train)

        # Évaluation sur le test
        y_pred    = pipeline.predict(X_test)
        accuracy  = accuracy_score(y_test, y_pred)
        print(f"Test accuracy        : {accuracy:.3f}")
        print(classification_report(y_test, y_pred, zero_division=0))

        results[model_name] = {
            'pipeline': pipeline,
            'accuracy': accuracy,
            'cv_mean':  cv_scores.mean(),
            'y_test':   y_test,
            'y_pred':   y_pred
        }

    return results, X_test, y_test

def baseline_rules(questions):
    import re
    CODE_RE = re.compile(r'\b[A-Z]{1,3}[-]?\d{1,4}\b', re.IGNORECASE)
    CHAT_TRIGGERS = {'bonjour', 'bonsoir', 'salut', 'hello', 'merci', 'au', 'revoir', 'aide', 'comment', 'qui', 'que'}
    QUESTION_WORDS = {'quel', 'quels', 'quelle', 'comment', 'pourquoi', 'quand', 'y'}

    predictions = []
    for q in questions:
        if CODE_RE.search(q):
            predictions.append('code_lookup')
        elif any(w in q.lower().split() for w in CHAT_TRIGGERS):
            predictions.append('chat')
        elif '?' in q or any(w in q.lower().split() for w in QUESTION_WORDS):
            predictions.append('semantic')
        else:
            predictions.append('keyword')
    return predictions

def plot_confusion_matrix(y_test, y_pred, model_name):
    labels = ['chat', 'code_lookup', 'keyword', 'semantic']
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f'Matrice de confusion — {model_name}')
    plt.tight_layout()
    plt.savefig(f'ml/intent_classifier/model/confusion_matrix_{model_name}.png', dpi=150)
    print(f"Matrice sauvegardée : ml/intent_classifier/model/confusion_matrix_{model_name}.png")
    plt.close()

def plot_comparison(results, y_test):
    models    = list(results.keys()) + ['baseline_rules']
    baseline_pred = baseline_rules(y_test.tolist())
    accuracies = [
        results['logistic']['accuracy'],
        results['svm']['accuracy'],
        accuracy_score(y_test, baseline_pred)
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, accuracies, color=['#2196F3', '#4CAF50', '#FF9800'])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Accuracy')
    ax.set_title('Comparaison : ML vs Règles')
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.2f}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('ml/intent_classifier/model/comparison_ml_vs_rules.png', dpi=150)
    print("Comparaison sauvegardée : ml/intent_classifier/model/comparison_ml_vs_rules.png")
    plt.close()

if __name__ == "__main__":
    print("=== ENTRAÎNEMENT DU CLASSIFIEUR D'INTENTION ===\n")

    # Étape 1 : Données
    df = load_data()
    df = add_features(df)

    # Étape 2 : Entraînement
    results, X_test, y_test = train_and_evaluate(df)

    # Étape 3 : Visualisations
    for model_name, res in results.items():
        plot_confusion_matrix(res['y_test'], res['y_pred'], model_name)
    plot_comparison(results, y_test)

    # Étape 4 : Sauvegarder le meilleur modèle
    best_name = max(results, key=lambda k: results[k]['accuracy'])
    best_pipeline = results[best_name]['pipeline']
    print(f"\nMeilleur modèle : {best_name} (accuracy={results[best_name]['accuracy']:.3f})")

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(best_pipeline, f)
    print(f"Modèle sauvegardé : {MODEL_PATH}")

    # Étape 5 : Test rapide
    print("\n=== TEST RAPIDE ===")
    test_questions = [
        "A100",
        "Ingection",
        "Quels actes pour les fractures ?",
        "bonjour à vous"
    ]
    for q in test_questions:
        pred  = best_pipeline.predict([q])[0]
        proba = best_pipeline.predict_proba([q])[0]
        conf  = max(proba)
        print(f"  '{q}' → {pred} (confiance: {conf:.2f})")