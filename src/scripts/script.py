# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:53:18 2024

@author: milli
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, ttest_ind
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Importazione dei dati
data = pd.read_csv('data.csv')

# 2. Calcolo delle metriche di concordanza

# Funzione per calcolare la media dei voti umani per ogni riga
data['Media Voti Umani'] = data[['Sasha', 'Mila', 'Ari']].mean(axis=1)

# Inizializza liste per memorizzare le correlazioni
ai_columns = ['gpt', 'cohreForAi', 'NousResearch', 'Meta_llama', 'Google', 'Mixtrail', 'Zephyr-orpo', 'Copilot', 'Mistral']
criteri = ['1','2','3','4','5','6','7','8','9','10']

# inizializzazioni
pearson_corrs = []
spearman_corrs = []
kappas = []
varianze = []

# Calcola le metriche di concordanza per ogni AI: la correlazione di Pearson e di Spearman
for col in ai_columns:
    pearson_corr, _ = pearsonr(data[col], data['Media Voti Umani'])
    spearman_corr, _ = spearmanr(data[col], data['Media Voti Umani'])
    
    # Per il kappa, bisogna trasformare i voti in categorie 
    kappa = cohen_kappa_score(data[col].round(), data['Media Voti Umani'].round())
    
    pearson_corrs.append(pearson_corr)
    spearman_corrs.append(spearman_corr)
    kappas.append(kappa)
    
    # Calcola la varianza
    varianza = np.var(data[col])
    varianze.append(varianza)

# Media delle correlazioni
mean_pearson = np.mean(pearson_corrs)
mean_spearman = np.mean(spearman_corrs)
mean_kappa = np.mean(kappas)

# visualizzazione delle correlazioni
print(f'Media Correlazione di Pearson: {mean_pearson}')
print(f'Media Correlazione di Spearman: {mean_spearman}')
print(f'Media Kappa di Cohen: {mean_kappa}\n')

# 3. Analisi delle differenze medie

# Media della differenza tra i voti degli umani e ciascun AI
differenze_medie = []
std_differenze_medie = []

for col in ai_columns:
    differenza_media = (data[col] - data['Media Voti Umani']).mean()
    std_differenza_media = (data[col] - data['Media Voti Umani']).std()
    differenze_medie.append(differenza_media)
    std_differenze_medie.append(std_differenza_media)
    
print(f'Differenze Medie: {differenze_medie}')
print(f'STD Differenze Medie: {std_differenze_medie}\n')

# 4. Analisi delle differenze di medie per ciascun criterio

# Media per ciascun criterio dei voti umani
mean_human_by_criterio = data.groupby('criterio')['Media Voti Umani'].mean()
dev_human_by_criterio = data.groupby('criterio')['Media Voti Umani'].mean()

differenza_media_criterio = []
differenza_std_criterio = []
for col in ai_columns:
    # differenza tra le medie per ciascun criterio
    differenza_media_criterio.append(data.groupby('criterio')[col].mean() - mean_human_by_criterio)
    differenza_std_criterio.append(data.groupby('criterio')[col].std() + dev_human_by_criterio)

# Crea un DataFrame dalle differenze calcolate
differenza_media_criterio_df = pd.DataFrame(differenza_media_criterio).T
differenza_media_criterio_df.columns = ai_columns

# Crea un DataFrame dalle deviazioni standard calcolate
differenza_std_criterio_df = pd.DataFrame(differenza_std_criterio).T
differenza_std_criterio_df.columns = ai_columns

# Calcola la media delle differenze medie di ciascun criterio per le AI
media_delle_medie_per_criterio = differenza_media_criterio_df.mean(axis=1)

#  deviazioni standard di ciascun criterio per le AI
media_delle_std_per_criterio = differenza_std_criterio_df.mean(axis=1)

print(f'Differenza media per criterio :\n {differenza_media_criterio_df}')
print(f'Media delle differenze medie per criterio:\n {media_delle_medie_per_criterio}')

# 5. Creazione di un grafico a barre delle differenze medie
plt.figure(figsize=(12, 6))
plt.bar(ai_columns, differenze_medie, color='skyblue', width=0.5)
plt.axhline(0, color='grey', linewidth=0.8)
plt.xlabel('AI')
plt.ylabel('Differenze Medie')
plt.title('Differenze Medie tra Voti AI e Media Voti Umani')
plt.show()

# 6. Box plot voti AI e voti umani
plt.figure(figsize=(10, 6))
data_melted = pd.melt(data, id_vars=['frase', 'criterio'], value_vars=ai_columns + ['Media Voti Umani'],
                      var_name='Valutatore', value_name='Voto')
sns.boxplot(x='Valutatore', y='Voto', data=data_melted)
plt.xticks(rotation=45)
plt.title('Box Plot Voti AI vs Media Voti Umani')
plt.show()

# 7. Visualizzazione tramite HeatMap

# Calcola la matrice delle correlazioni
correlation_matrix = data[ai_columns + ['Media Voti Umani']].corr()

# Visualizza la heatmap delle correlazioni
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
plt.title('Heatmap delle Correlazioni tra AI e Media Voti Umani')
plt.show()

# Salva i risultati in un file CSV (commentato)
# data.to_csv('path/to/save/analisi_output.csv', index=False)
