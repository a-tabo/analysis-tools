import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#load data
spi_files = [
	"lco-104_aspi-1-1x_ads.txt",
	"lco-104_aspi-2-1x_ads.txt",
	"lco-104_aspi-3-1x_ads.txt",
	"lco-104_aspi-4-1x_ads.txt",
	"lco-104_aspi-8-1x_ads.txt",
	"lco-104_aspi-9-1x_ads.txt",
]
spi_data = [np.genfromtxt(file, skip_header=1) for file in spi_files]

#extract configuration and energies
conf = spi_data[0][:8, 0]
data = {f"ASPI {i+1}": spi[:8, 1] for i, spi in enumerate(spi_data)}

#flatten energies for clustering
all_energies = np.concatenate(list(data.values()))

#perform KMeans clustering
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(all_energies.reshape(-1, 1))

#plot results
cluster_colors = ['red', 'blue', 'green']
cluster_labels = kmeans.labels_

fig, ax1 = plt.subplots(figsize=(8, 6))

#scatter plot with clustering
for idx, (label, values) in enumerate(data.items()):
    cluster_indices = labels[idx * len(conf):(idx + 1) * len(conf)]
    for i, energy in enumerate(values):
        cluster_label = f'Cluster {cluster_indices[i] + 1}'
        if cluster_label not in ax1.get_legend_handles_labels()[1]:
            ax1.scatter(
                conf[i], energy, color=cluster_colors[cluster_indices[i]], label=cluster_label
                )
        else:
            ax1.scatter(conf[i], energy, color=cluster_colors[cluster_indices[i]])

#plot centroids
centroids = kmeans.cluster_centers_
for i, centroid in enumerate(centroids):
    ax1.axhline(
        centroid, color=cluster_colors[i], linestyle='--', label=f'Centroid {i + 1}'
    )

#customize plot
ax1.set_title('LCO ASPI-Li Ads. Conf.', fontsize=20)
ax1.set_xlabel('Ads. Conf.', fontsize=14)
ax1.set_xlim(0, 9)
ax1.set_xticks(range(0, 9, 2))
ax1.set_ylabel('Ads. Ener. [eV]', fontsize=14)
ax1.set_ylim(-4, 0)
ax1.tick_params(axis='both', labelsize=12)
ax1.legend(fontsize=10)

#save plot
plt.tight_layout()
plt.savefig('ads-conf_1-8_plot.png')
plt.show()

#print centroids values
print("Centroids:", centroids)