import torch
from sklearn.cluster import KMeans


class KMeansClusterFinder:
    def __init__(self, data, n_clusters):
        self.data = data
        self.n_clusters = n_clusters

    def get_mask(self, nc_curves):
        sampled_points = torch.stack([nc_curves[p] for p in self.data.curve_points], dim=0)
        sampled_points = sampled_points.permute(1, 2, 3, 0)
        sampled_points = sampled_points[self.data.class_idx]
        H, W = sampled_points.shape[:2]
        reshaped_points = sampled_points.reshape(-1, sampled_points.shape[-1])

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        labels = kmeans.fit_predict(reshaped_points.cpu().numpy())

        kandinsky_mask = labels.reshape(H, W)

        return kandinsky_mask
