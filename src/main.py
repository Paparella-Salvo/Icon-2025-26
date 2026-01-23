import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from data_processing import WeatherDataLoader
from supervised_models import SupervisedManager
from belief_network import BeliefNetworkManager

# ============================
# Funzione per etichette barre
# ============================
def autolabel(rects, decimals=4):
    """Aggiunge il valore numerico sopra ogni barra."""
    for rect in rects:
        height = rect.get_height()
        label = f"{height:.{decimals}f}" if decimals else f"{int(height)}"
        plt.text(
            rect.get_x() + rect.get_width() / 2.,
            height + (0.01 if decimals else 0.3),
            label,
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

def main():
    # Gestione dati
    loader = WeatherDataLoader()
    loader.load_and_clean_data()
    loader.apply_mapping()
    loader.eda_visualization()

    # Apprendimento Supervisionato
    X_train, y_train, X_test, y_test = loader.prepare_supervised_data()
    print("\n=== START SUPERVISED LEARNING ===")
    df_full = loader.df_mapped.drop(columns=['Weather Type']) 
    y_full = loader.df_mapped['Weather Type']
    supervised_suite = SupervisedManager()
    supervised_suite.run_all_models(df_full, y_full)

    # Belief Network
    print("\n=== START BELIEF NETWORK ===")
    
    df_belief = loader.prepare_belief_data()

    # --- K2 ---
    bn_manager_k2 = BeliefNetworkManager(df_belief)
    bn_manager_k2.learn_structure(scoring_method='bayesian')
    bn_manager_k2.visualize("Bayesian Network - Search & Score (Bayesian)")
    acc_k2 = bn_manager_k2.evaluate()

    # --- BIC ---
    bn_manager_bic = BeliefNetworkManager(df_belief)
    bn_manager_bic.learn_structure(scoring_method='bic')
    bn_manager_bic.visualize("Bayesian Network - BIC")
    acc_bic = bn_manager_bic.evaluate()

    # --- Expert ---
    expert_edges = [
        ('Season', 'Temperature'),
        ('Season', 'Humidity'),
        ('Location', 'Temperature'),
        ('Location', 'Humidity'),
        ('Atmospheric Pressure', 'Cloud Cover'),
        ('Temperature', 'Weather Type'),
        ('Humidity', 'Weather Type'),
        ('Wind Speed', 'Weather Type'),
        ('Cloud Cover', 'Weather Type')
    ]

    bn_manager_exp = BeliefNetworkManager(df_belief)
    bn_manager_exp.set_expert_structure(expert_edges)
    bn_manager_exp.visualize("Bayesian Network - Expert Structure")
    acc_exp = bn_manager_exp.evaluate()

    # ============================
    # RIEPILOGO ACCURACY
    # ============================
    print("\n--- RIEPILOGO ACCURACY BN ---")
    print(f"K2:     {acc_k2:.4f}")
    print(f"BIC:    {acc_bic:.4f}")
    print(f"Expert: {acc_exp:.4f}")

    plt.figure(figsize=(8, 6))
    rects = plt.bar(['K2', 'BIC', 'Expert'],
                    [acc_k2, acc_bic, acc_exp],
                    color=['#1F3A5F', '#5C7AEA', '#A3B18A'])

    plt.title('Confronto Accuratezza Reti Bayesiane')
    plt.ylabel('Accuracy')
    plt.ylim(0.7, 1.0)
    autolabel(rects, decimals=4)
    plt.show()

    # ============================
    # CONFRONTO NUMERO ARCHI
    # ============================
    edges_k2 = bn_manager_k2.count_edges()
    edges_bic = bn_manager_bic.count_edges()
    edges_exp = bn_manager_exp.count_edges()

    plt.figure(figsize=(8, 6))
    rects = plt.bar(['K2', 'BIC', 'Expert'],
                    [edges_k2, edges_bic, edges_exp],
                    color=['#1F3A5F', '#5C7AEA', '#A3B18A'])

    plt.title('Numero Archi nelle Bayesian Network')
    plt.ylabel('Numero Archi')
    autolabel(rects, decimals=0)
    plt.show()

if __name__ == "__main__":
    main()
