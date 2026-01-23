import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold #GridSearchCV
from sklearn.metrics import (
    make_scorer,
    f1_score,
    classification_report,
    balanced_accuracy_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# ==========================================================
# Funzioni Helper – Apprendimento Supervisionato (da esame)
# ==========================================================

def train_model_simple(model_class, param_list, X_train, y_train, cv_folds=5, random_state=42):
    """
    Selezione del modello tramite K-Fold Cross-Validation.
    Gli iperparametri vengono valutati manualmente.
    """
    scorer = make_scorer(f1_score, average='weighted')
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    best_score = -np.inf
    best_model = None

    for params in param_list:
        model = model_class(**params)
        scores = []

        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            scores.append(f1_score(y_val, y_pred, average='weighted'))

        mean_score = np.mean(scores)

        if mean_score > best_score:
            best_score = mean_score
            best_model = model

    return best_model


def evaluate_model(model, X_test, y_test):
    """
    Valutazione finale sul Test Set.
    """
    y_pred = model.predict(X_test)

    return {
        'accuracy': balanced_accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }


# ==========================================================
# Classe di gestione dei modelli supervisionati
# ==========================================================

class SupervisedManager:
    def __init__(self):
        self.stats = {}
        self.summary = {}

    # ------------------------------------------------------
    # Pipeline principale: 5 run con 5 split diversi
    # ------------------------------------------------------
    def run_all_models(self, df_full, y_full, n_runs=5):

        # Struttura dati per salvare i risultati
        self.stats = {
            'KNN': {'accuracy': [], 'f1': []},
            'DecisionTree': {'accuracy': [], 'f1': []},
            'RandomForest': {'accuracy': [], 'f1': []},
            'SVM': {'accuracy': [], 'f1': []}
        }

        for i in range(n_runs):
            print(f"\n=== RUN {i+1} ===")

            # Nuovo split per tutti i modelli
            X_train, X_test, y_train, y_test = train_test_split(
                df_full, y_full, test_size=0.2, stratify=y_full, random_state=i
            )

            # ------------------ KNN ------------------
            knn_model = train_model_simple(
                KNeighborsClassifier,
                [
                    {'n_neighbors': 5, 'weights': 'uniform', 'n_jobs': -1},
                    {'n_neighbors': 7, 'weights': 'distance', 'n_jobs': -1}
                ],
                X_train, y_train
            )
            knn_eval = evaluate_model(knn_model, X_test, y_test)
            self.stats['KNN']['accuracy'].append(knn_eval['accuracy'])
            self.stats['KNN']['f1'].append(knn_eval['f1'])

            # ---------------- Decision Tree ----------------
            dt_model = train_model_simple(
                DecisionTreeClassifier,
                [
                    {'max_depth': 5, 'min_samples_split': 2, 'random_state': 42},
                    {'max_depth': 10, 'min_samples_split': 2, 'random_state': 42}
                ],
                X_train, y_train
            )
            dt_eval = evaluate_model(dt_model, X_test, y_test)
            self.stats['DecisionTree']['accuracy'].append(dt_eval['accuracy'])
            self.stats['DecisionTree']['f1'].append(dt_eval['f1'])

            # ---------------- Random Forest ----------------
            rf_model = train_model_simple(
                RandomForestClassifier,
                [
                    {'n_estimators': 50, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1},
                    {'n_estimators': 100, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1}
                ],
                X_train, y_train
            )
            rf_eval = evaluate_model(rf_model, X_test, y_test)
            self.stats['RandomForest']['accuracy'].append(rf_eval['accuracy'])
            self.stats['RandomForest']['f1'].append(rf_eval['f1'])

            # ---------------- SVM ----------------
            svm_model = train_model_simple(
                SVC,
                [
                    {'C': 1, 'kernel': 'rbf'},
                    {'C': 10, 'kernel': 'rbf'}
                ],
                X_train, y_train
            )
            svm_eval = evaluate_model(svm_model, X_test, y_test)
            self.stats['SVM']['accuracy'].append(svm_eval['accuracy'])
            self.stats['SVM']['f1'].append(svm_eval['f1'])
       
        # Calcolo medie e std
        self._compute_summary()
        self._print_summary()
        # Grafici
        self._plot_all_model_runs()
        self._plot_comparison()

    
    def _print_summary(self):
        print("\n=== RISULTATI MEDI DEI MODELLI (5 RUN) ===")
        for model, stats in self.summary.items():
            print(f"\n--- {model} ---")
            print(f"Accuracy media: {stats['accuracy_mean']:.4f}")
            print(f"F1-score medio: {stats['f1_mean']:.4f}")


    # ------------------------------------------------------
    # Calcolo medie e deviazioni standard
    # ------------------------------------------------------
    def _compute_summary(self):
        self.summary = {}
        for model in self.stats:
            acc = self.stats[model]['accuracy']
            f1 = self.stats[model]['f1']
            self.summary[model] = {
                'accuracy_mean': np.mean(acc),
                'f1_mean': np.mean(f1),
            }


    # ------------------------------------------------------
    # Grafico dei 5 run per ogni modello
    # ------------------------------------------------------
    def _plot_all_model_runs(self):
        for model in self.stats:
            runs = np.arange(1, len(self.stats[model]['accuracy']) + 1)

            # ORIZZONTALE
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

            # VERTICALE
            #fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

            # --- Grafico Accuracy ---
            axes[0].plot(runs, self.stats[model]['accuracy'], marker='o', color='blue')
            axes[0].set_title(f"Accuracy dei 5 run – {model}")
            axes[0].set_ylabel("Accuracy")
            axes[0].set_ylim(0.7, 1.0)
            axes[0].grid(True, linestyle='--', alpha=0.6)

            # --- Grafico F1 ---
            axes[1].plot(runs, self.stats[model]['f1'], marker='o', color='orange')
            axes[1].set_title(f"F1-score dei 5 run – {model}")
            axes[1].set_xlabel("Run")
            axes[1].set_ylabel("F1-score")
            axes[1].set_ylim(0.7, 1.0)
            axes[1].grid(True, linestyle='--', alpha=0.6)

            plt.tight_layout()
            plt.show()



    # ------------------------------------------------------
    # Grafico finale con medie ± std
    # ------------------------------------------------------
    def _plot_comparison(self):
        stats_df = pd.DataFrame(self.summary).T

        plt.figure(figsize=(12, 8))
        x = np.arange(len(stats_df))
        width = 0.35

        rects1 = plt.bar(x - width/2, stats_df['accuracy_mean'], width, label='Accuracy', color='tab:blue', alpha=0.7)
        rects2 = plt.bar(x + width/2, stats_df['f1_mean'], width, label='F1 Score', color='tab:orange', alpha=0.7)

        def autolabel(rects): 
            for rect in rects: 
                height = rect.get_height() 
                plt.text(rect.get_x() + rect.get_width()/2., height + 0.002, 
                         f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold') 
        autolabel(rects1) 
        autolabel(rects2)

        plt.xticks(x, stats_df.index)
        plt.ylabel('Score')
        plt.title('Confronto Prestazioni Medie')
        plt.ylim(0.80, 1.0)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
    

