import os
import kagglehub
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

class WeatherDataLoader:
    def __init__(self):
        self.df = None
        self.df_mapped = None
        self.X_train_resampled = None
        self.y_train_resampled = None
        self.X_test_scaled = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.column_names = None # preservare i nomi dopo lo scaling

    def load_and_clean_data(self):
        print("--- Download e Caricamento Dati ---")
        path = kagglehub.dataset_download("nikhil7280/weather-type-classification")
        csv_path = None
        for filename in os.listdir(path):
            if filename.endswith(".csv"):
                csv_path = os.path.join(path, filename)
                break
        
        if csv_path is None:
            raise FileNotFoundError("Nessun file CSV trovato nel dataset scaricato.")

        self.df = pd.read_csv(csv_path)
        
        # Rimozione duplicati e nulli
        if self.df.duplicated().sum() > 0:
            self.df.drop_duplicates(inplace=True)
        if self.df.isnull().sum().sum() > 0:
            self.df.dropna(inplace=True)
            
        print(f"Dataset caricato: {self.df.shape[0]} righe, {self.df.shape[1]} colonne.")
        return self.df

    def apply_mapping(self):
        # Mapping
        map_cloud = {'clear': 0, 'partly cloudy': 1, 'cloudy': 2, 'overcast': 3}
        map_season = {'Spring': 0, 'Summer': 1, 'Autumn': 2, 'Winter': 3}
        map_location = {'coastal': 0, 'inland': 1, 'mountain': 2}
        map_weather = {'Sunny': 0, 'Cloudy': 1, 'Rainy': 2, 'Snowy': 3}

        self.df_mapped = self.df.copy()
        self.df_mapped['Cloud Cover'] = self.df_mapped['Cloud Cover'].map(map_cloud)
        self.df_mapped['Season'] = self.df_mapped['Season'].map(map_season)
        self.df_mapped['Location'] = self.df_mapped['Location'].map(map_location)
        self.df_mapped['Weather Type'] = self.df_mapped['Weather Type'].map(map_weather)
        
        return self.df_mapped

    def eda_visualization(self):
        print("--- Generazione Grafici EDA ---")

        # 1. Matrice di correlazione
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.df_mapped.corr(), cmap="coolwarm", annot=True, fmt=".2f")
        plt.title("Matrice di Correlazione")
        plt.tight_layout()
        plt.show()

        # 2. Istogrammi + KDE per feature numeriche
        cols_numeric = [
            'Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)',
            'Atmospheric Pressure', 'UV Index', 'Visibility (km)'
        ]
        for col in cols_numeric:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.df[col], kde=True, bins=35, color='g')
            plt.title(f'Distribuzione di {col}')
            plt.xlabel(col)
            plt.ylabel('Frequenza')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()

        # 3. Countplot per feature categoriche rispetto al target
        cols_categorical = ['Cloud Cover', 'Season', 'Location']
        palette = ["#d0e1f9", "#a6bddb", "#3690c0", "#034e7b"]  # chiaro → scuro

        for col in cols_categorical:
            plt.figure(figsize=(12, 8))

            #ordinamento dei vaolori di Cloud Cover in base all'intesità dela palette di colori 
            if col == 'Cloud Cover':
                hue_order = ['clear', 'partly cloudy', 'cloudy', 'overcast']
                sns.countplot(
                    x="Weather Type",
                    hue=col,
                    data=self.df,
                    palette=palette,
                    hue_order=hue_order
                )
            else:
                sns.countplot(
                    x="Weather Type",
                    hue=col,
                    data=self.df,
                    palette=palette
                )

            plt.title(f'Relazione tra {col} e Weather Type')
            plt.xlabel("Weather Type")
            plt.ylabel("Conteggio")
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()



    def prepare_supervised_data(self):
        print("--- Preparazione Dati per Supervised Learning ---")
        X = self.df_mapped.drop('Weather Type', axis=1)
        y = self.df_mapped['Weather Type']
        self.column_names = X.columns

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        self.y_test = y_test

        # Scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = pd.DataFrame(self.scaler.transform(X_test), columns=self.column_names)
        
        # SMOTE
        print("Applicazione SMOTE...")
        try:
            smote = SMOTE(random_state=42, k_neighbors=3)
            self.X_train_resampled, self.y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
            # Riconversione in DataFrame per mantenere i nomi delle colonne (utile per Feature Importance)
            self.X_train_resampled = pd.DataFrame(self.X_train_resampled, columns=self.column_names)
        except ValueError as e:
            print(f"Errore SMOTE: {e}. Uso dati non bilanciati.")
            self.X_train_resampled = pd.DataFrame(X_train_scaled, columns=self.column_names)
            self.y_train_resampled = y_train

        return self.X_train_resampled, self.y_train_resampled, self.X_test_scaled, self.y_test

    def prepare_belief_data(self):
        # Discretizzazione specifica per Belief Network
        print("--- Preparazione Dati per Belief Network ---")
        discretize_info = {
            'Temperature': {'bins': [-50, 0, 10, 20, 30, 120], 'labels': ['very_cold', 'cold', 'mild', 'warm', 'hot']},
            'Humidity': {'bins': [0, 20, 40, 60, 80, 120], 'labels': ['very_low', 'low', 'medium', 'high', 'very_high']},
            'Wind Speed': {'bins': [0, 10, 20, 30, 50, 200], 'labels': ['calm', 'breeze', 'moderate', 'strong', 'storm']},
            'Atmospheric Pressure': {'bins': [800, 990, 1010, 1025, 1040, 1200], 'labels': ['very_low', 'low', 'normal', 'high', 'very_high']},
            'Precipitation (%)': {'bins': [-1, 0, 20, 60, 120], 'labels': ['none', 'low', 'moderate', 'high']},
            'UV Index': {'bins': [-1, 2, 5, 7, 10, 15], 'labels': ['low', 'moderate', 'high', 'very_high', 'extreme']},
            'Visibility (km)': {'bins': [-1, 2, 5, 10, 25], 'labels': ['fog', 'haze', 'clear', 'very_clear']}
        }

        df_belief = self.df.copy() # Utilizzo del df originale non mappato numericamente 
        
        for col, info in discretize_info.items():
            df_belief[col] = pd.cut(df_belief[col], bins=info['bins'], labels=info['labels'], include_lowest=True)
            
        return df_belief