import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, K2Score, BicScore, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class BeliefNetworkManager:
    def __init__(self, df):
        # Mantenimento del dataframe originale discretizzato
        self.df = df
        # Split 80/20
        self.train_data, self.test_data = train_test_split(df, test_size=0.2, random_state=42)
        self.model = None
        self.inference = None

    def learn_structure(self, scoring_method='bayesian'):
        """
        Apprende la struttura dai dati tramite approccio Search & Score per Reti Bayesiane.
        """
        print(f"\n--- Learning Structure (Search & Score: {scoring_method.upper()} ---")
        
        # Scelta dello Scorer
        if scoring_method == 'bayesian':
            # Score bayesiano (approccio Search & Score)
            scoring = K2Score(self.train_data)
        elif scoring_method == 'bic':
            # Bayesian Information Criterion
            scoring = BicScore(self.train_data)
        else:
            raise ValueError("Metodo non supportato.")

        # Ricerca della struttura (Hill Climbing)
        est = HillClimbSearch(self.train_data)
        best_model = est.estimate(scoring_method=scoring, show_progress=True)
        
        # Creazione del Modello e Fitting dei parametri
        self.model = BayesianNetwork(best_model.edges())
        self.model.fit(self.train_data, estimator=MaximumLikelihoodEstimator)
        self.inference = VariableElimination(self.model)

    def set_expert_structure(self, expert_edges):
        """
        Imposta una struttura manuale (Expert).
        """
        print("\n--- Setting Expert Structure ---")
        self.model = BayesianNetwork(expert_edges)
        self.model.fit(self.train_data, estimator=MaximumLikelihoodEstimator)
        self.inference = VariableElimination(self.model)

    def visualize(self, title_text):
        """
        Mostra il grafo con il titolo specifico richiesto.
        """
        if self.model is None:
            print("Nessun modello da visualizzare.")
            return

        plt.figure(figsize=(12, 8))
        G = nx.DiGraph(self.model.edges())
        
        # Layout del grafo
        pos = nx.spring_layout(G, k=0.5, iterations=20)
        
        nx.draw(
            G, pos, 
            with_labels=True, 
            node_size=3000, 
            node_color="lightblue", 
            font_size=10, 
            font_weight="bold", 
            arrows=True,
            arrowsize=20
        )
        plt.title(title_text, fontsize=15)
        print(f"Generazione immagine: {title_text} (Chiudi la finestra per continuare)")
        plt.show()

    def evaluate(self, target='Weather Type'):
        """
        Valuta l'accuratezza del modello corrente.
        """
        if self.model is None:
            return 0.0

        y_true = self.test_data[target].values
        y_pred = []
        valid_nodes = set(self.model.nodes())

        # Iterezione su ogni riga del test set
        for _, row in self.test_data.iterrows():
            evidence = row.drop(target).to_dict()
            # Filtro solo sui nodi che esistono davvero nel grafo appreso
            evidence = {k: v for k, v in evidence.items() if k in valid_nodes}
            
            try:
                # Predizione (Map Query)
                pred = self.inference.map_query(variables=[target], evidence=evidence, show_progress=False)
                y_pred.append(pred[target])
            except:
                y_pred.append(None) # Gestione caso errore inferenza

        # Pulizia per calcolo metriche (rimuozione i None e converte in stringa)
        clean_true = []
        clean_pred = []
        for t, p in zip(y_true, y_pred):
            if p is not None:
                clean_true.append(str(t))
                clean_pred.append(str(p))
        
        if len(clean_pred) == 0:
            print("Nessuna predizione valida effettuata.")
            return 0.0

        acc = accuracy_score(clean_true, clean_pred)
        print(f"Accuracy [{target}]: {acc:.4f}")
        return acc
    
    def count_edges(self):
        if self.model is None:
            return 0
        return len(self.model.edges())
