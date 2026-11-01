import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from vectorstore.chroma_client import get_collection

def get_embeddings_from_chromadb():
    collection = get_collection()
    
    if collection.count() == 0:
        raise ValueError("ChromaDB est vide — lance l'ingestion d'abord.")
    
    results = collection.get(include=['embeddings', 'documents', 'metadatas'])
    
    embeddings = np.array(results['embeddings'])
    documents  = results['documents']
    metadatas  = results['metadatas']
    
    print(f"Embeddings récupérés : {embeddings.shape}")
    print(f"  → {embeddings.shape[0]} chunks, {embeddings.shape[1]} dimensions")
    
    return embeddings, documents, metadatas

def find_optimal_k(embeddings, k_max=8):
    inertias       = []
    silhouettes    = []
    k_values       = range(2, min(k_max + 1, len(embeddings)))

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(embeddings, labels))
        print(f"  K={k} | inertie={km.inertia_:.1f} | silhouette={silhouette_score(embeddings, labels):.3f}")

    # Graphique méthode du coude
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(list(k_values), inertias, 'bo-')
    ax1.set_xlabel('Nombre de clusters (K)')
    ax1.set_ylabel('Inertie')
    ax1.set_title('Méthode du coude')
    ax1.grid(True)

    ax2.plot(list(k_values), silhouettes, 'ro-')
    ax2.set_xlabel('Nombre de clusters (K)')
    ax2.set_ylabel('Score silhouette')
    ax2.set_title('Score silhouette par K')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('ml/clustering/output/elbow_curve.png', dpi=150)
    print("Graphique sauvegardé : ml/clustering/output/elbow_curve.png")
    plt.close()

    # Meilleur K = celui avec le plus haut score silhouette
    best_k = list(k_values)[silhouettes.index(max(silhouettes))]
    print(f"\nMeilleur K recommandé : {best_k}")
    return best_k

def cluster_and_visualize(embeddings, metadatas, n_clusters):
    # K-Means
    km     = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)

    # Réduction PCA → 2 dimensions pour visualisation
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)

    print(f"\nVariance expliquée par PCA : {sum(pca.explained_variance_ratio_)*100:.1f}%")

    # Visualisation
    colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
    fig, ax = plt.subplots(figsize=(10, 7))

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[colors[cluster_id]],
            label=f'Cluster {cluster_id + 1}',
            s=80, alpha=0.7
        )
        # Annoter avec le code médical
        for i, (x, y) in enumerate(coords[mask]):
            code = metadatas[np.where(mask)[0][i]].get('code', '')
            if code:
                ax.annotate(code, (x, y),
                           textcoords="offset points",
                           xytext=(5, 5), fontsize=7)

    ax.set_title(f'Clustering K-Means ({n_clusters} clusters) — PCA 2D')
    ax.set_xlabel(f'PCA 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PCA 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ml/clustering/output/clusters_visualisation.png', dpi=150)
    print("Visualisation sauvegardée : ml/clustering/output/clusters_visualisation.png")
    plt.close()

    return labels

def analyze_clusters(labels, metadatas, n_clusters):
    print(f"\n{'='*50}")
    print("ANALYSE DES CLUSTERS")
    print(f"{'='*50}")

    for cluster_id in range(n_clusters):
        indices = [i for i, l in enumerate(labels) if l == cluster_id]
        print(f"\nCluster {cluster_id + 1} ({len(indices)} actes) :")

        categories = set()
        codes      = []
        for i in indices:
            meta = metadatas[i]
            if meta.get('code'):
                codes.append(meta['code'])
            if meta.get('category'):
                categories.add(meta['category'][:50])

        print(f"  Codes    : {', '.join(codes[:10])}")
        print(f"  Catégories : {', '.join(list(categories)[:3])}")

if __name__ == "__main__":
    print("=== CLUSTERING DES ACTES MÉDICAUX ===\n")

    # Étape 1 : Récupérer les données
    embeddings, documents, metadatas = get_embeddings_from_chromadb()

    # Étape 2 : Trouver le bon K
    print("\n--- Recherche du K optimal ---")
    best_k = find_optimal_k(embeddings, k_max=6)

    # Étape 3 : Clustering + visualisation
    print(f"\n--- Clustering avec K={best_k} ---")
    labels = cluster_and_visualize(embeddings, metadatas, best_k)

    # Étape 4 : Analyse
    analyze_clusters(labels, metadatas, best_k)

    print("\n✅ Clustering terminé !")
    print("Résultats dans : ml/clustering/output/")