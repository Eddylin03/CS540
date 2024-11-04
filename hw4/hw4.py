import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import heapq

def load_data(filepath):
    data = []

    with open(filepath, 'r') as file:
        # DictReader 
        reader = csv.DictReader(file)
        
        for row in reader:
            # Convert to a regular dict
            row_dict = dict(row)
            # Append the row dict
            data.append(row_dict)

    return data

def calc_features(row):

    features = []
    
    # List of keys in order
    keys = ['Population', 'Net migration', 'GDP ($ per capita)', 
            'Literacy (%)', 'Phones (per 1000)', 'Infant mortality (per 1000 births)']
    
    for key in keys:
        value_str = row[key]
        value_str = value_str.replace(',', '').replace('%', '').strip()
        # Handle missing values
        if value_str in ['', 'N/A', 'unknown']:
            value = 0.0 
        else:
            value = float(value_str)

        features.append(value)
    

    feature_array = np.array(features, dtype=np.float64)
    
    return feature_array


import numpy as np
import heapq

def hac(features):
    n = len(features)
    Z = np.zeros((n - 1, 4))
    current_cluster_index = n 
    clusters = {i: 1 for i in range(n)} 

    # Compute initial distances and initialize the heap
    heap = []
    D = {}
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(features[i] - features[j])
            D[frozenset({i, j})] = dist
            heapq.heappush(heap, (dist, i, j))

    # Active cluster indices: this is in the help of GPT 4o1-preview
    active_clusters = set(range(n))

    for iteration in range(n - 1):
        while True:
            # Pop the smallest distance from the heap
            dist, c1, c2 = heapq.heappop(heap)

            # Check if both clusters are still active
            if c1 in active_clusters and c2 in active_clusters:
                break  # Valid pair found

        idx1, idx2 = min(c1, c2), max(c1, c2)  # For tie-breaking
        Z[iteration] = [idx1, idx2, dist, clusters[c1] + clusters[c2]]

        # Update clusters
        clusters[current_cluster_index] = clusters[c1] + clusters[c2]
        active_clusters.remove(c1)
        active_clusters.remove(c2)
        active_clusters.add(current_cluster_index)

        # Update distances
        for k in active_clusters:
            if k != current_cluster_index:

                d1 = D.get(frozenset({c1, k}), np.inf)
                d2 = D.get(frozenset({c2, k}), np.inf)
                new_dist = min(d1, d2)

                D[frozenset({current_cluster_index, k})] = new_dist

                # Add new distance to the heap
                heapq.heappush(heap, (new_dist, current_cluster_index, k))

                # Remove old distances
                D.pop(frozenset({c1, k}), None)
                D.pop(frozenset({c2, k}), None)

        # Increment the cluster index
        current_cluster_index += 1

    return Z




def fig_hac(Z, names):
    # Import necessary functions
    from scipy.cluster.hierarchy import dendrogram
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 7))
    
    dendrogram(Z, labels=names, leaf_rotation=90)
    
    plt.tight_layout()
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Countries')
    plt.ylabel('Distance')
    plt.show()
    return fig
def normalize_features(features):

    feature_matrix = np.vstack(features) 
    normalized_matrix = np.zeros_like(feature_matrix)
    for i in range(feature_matrix.shape[1]):
        col = feature_matrix[:, i]
        col_min = np.min(col)
        col_max = np.max(col)
        range_ = col_max - col_min
        
        if range_ == 0:
            # If the feature has zero range, set normalized values to zero
            normalized_col = np.zeros_like(col)
        else:
            #  normalization
            normalized_col = (col - col_min) / range_

        normalized_matrix[:, i] = normalized_col
    
    # Split the normalized matrix back into a list of arrays
    normalized_features = [normalized_matrix[i, :] for i in range(normalized_matrix.shape[0])]
    
    return normalized_features

if __name__ == "__main__":
    # Load and process data
    data = load_data('countries.csv')
    country_names = [row['Country'] for row in data]
    features = [calc_features(row) for row in data]
    features_normalized = normalize_features(features)
    n = 20
    selected_features = features_normalized[:n]
    selected_names = country_names[:n]

    Z = hac(selected_features)
    fig = fig_hac(Z, selected_names)
    plt.show()