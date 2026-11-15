from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
FEATURES = ['latitude', 'longitude', 'price', 'reviews_per_month', 'availability_365']
ANGLES = [45, 90, 180, 135, 0, 45]

listings = pd.read_csv(ROOT / 'data/paris_listings.csv')
features = listings[FEATURES].copy()
features[['price', 'reviews_per_month']] = np.log(features[['price', 'reviews_per_month']])
scaled = StandardScaler().fit_transform(features)
labels = DBSCAN(eps=0.25, min_samples=10).fit_predict(scaled)
counts = pd.Series(labels).value_counts()
clusters = [int(cluster) for cluster, count in counts.items() if cluster >= 0 and count >= 200]
clusters.sort(key=lambda cluster: listings.loc[labels == cluster, 'price'].median())
if len(clusters) != len(ANGLES):
    raise ValueError(f'Expected six letter clusters, found {len(clusters)}')
positions = {cluster: position for position, cluster in enumerate(clusters, 1)}
assigned = listings.assign(cluster=labels)
assigned['letter_position'] = assigned.cluster.map(positions).fillna(0).astype(int)
(ROOT / 'results').mkdir(exist_ok=True)
assigned.to_csv(ROOT / 'results/labels.csv', index=False)
summary = assigned[assigned.letter_position > 0].groupby('letter_position').agg(
    cluster=('cluster', 'first'),
    count=('cluster', 'size'),
    median_price=('price', 'median'),
)
summary.to_csv(ROOT / 'results/cluster_summary.csv')
figure, axes = plt.subplots(1, len(clusters), figsize=(18, 4), squeeze=False)
for position, (cluster, angle) in enumerate(zip(clusters, ANGLES)):
    local = PCA(n_components=2).fit_transform(scaled[labels == cluster])
    radians = np.deg2rad(angle)
    rotation = np.array([
        [np.cos(radians), -np.sin(radians)],
        [np.sin(radians), np.cos(radians)],
    ])
    aligned = local @ rotation
    axis = axes[0, position]
    axis.scatter(aligned[:, 0], aligned[:, 1], s=5, color='#264e73')
    axis.set_aspect('equal')
    axis.axis('off')
    price = listings.loc[labels == cluster, 'price'].median()
    axis.set_title(f'{position + 1} · median €{price:.2f}', fontsize=11)
figure.suptitle('Letter clusters ordered by increasing median nightly price', fontsize=17)
figure.tight_layout()
figure.savefig(ROOT / 'results/word.png', dpi=180)
plt.close(figure)
