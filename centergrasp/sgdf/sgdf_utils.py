import numpy as np
from scipy.spatial import KDTree


def argmin_n(ary, n):
    """Returns the indices of the n smallest elements from a numpy array."""
    indices = np.argpartition(ary, n)[:n]
    values = ary[indices]
    return indices, values


def argmax_n(ary, n):
    """Returns the indices of the n largest elements from a numpy array."""
    indices = np.argpartition(ary, -n)[-n:]
    values = ary[indices]
    return indices, values


def find_closest_vectorized(queries, candidates):
    """
    Find the closest candidate for each query point, and return the indeces.
    Input arrays must be np arrays of shape (q, 3) and (c, 3) respectively.
    This vectorized function is fast but memory intensive.
    """
    diff = candidates[None, ...] - queries[:, None, ...]
    dist = np.linalg.norm(diff, axis=-1)
    min_idxs = np.argmin(dist, axis=-1)
    min_vals = dist[np.arange(len(min_idxs)), min_idxs]
    return min_idxs, min_vals


def find_closest_chunked(queries, candidates, chunk_size=5000):
    """
    Find the closest candidate for each query point, and return the indices.
    Input arrays must be np arrays of shape (q, 3) and (c, 3) respectively.
    """
    min_idxs = np.zeros(len(queries), dtype=int)
    min_vals = np.zeros(len(queries))
    for i in range(0, len(queries), chunk_size):
        end = min(i + chunk_size, len(queries))
        chunk_queries = queries[i:end]
        chunk_diff = candidates[None, ...] - chunk_queries[:, None, ...]
        chunk_dist = np.linalg.norm(chunk_diff, axis=-1)
        chunk_min_idxs = np.argmin(chunk_dist, axis=-1)
        chunk_min_vals = chunk_dist[np.arange(len(chunk_min_idxs)), chunk_min_idxs]
        min_idxs[i:end] = chunk_min_idxs
        min_vals[i:end] = chunk_min_vals
    return min_idxs, min_vals


def knn(queries, candidates, k=1):
    """
    Find the k closest candidates for each query point, and return the indices.
    Input arrays must be np arrays of shape (q, 3) and (c, 3) respectively.
    """
    tree = KDTree(candidates)
    min_dists, min_idxs = tree.query(queries, k=k, workers=-1)
    return min_idxs, min_dists


def find_closest_unique(queries, candidates, max_dist=0.005):
    """
    Find the closest candidate for each query point, and return the unique indeces.
    Input arrays must be np arrays of shape (q, 3) and (c, 3) respectively.
    """
    min_idxs, min_vals = find_closest_chunked(queries, candidates)
    min_idxs_filtered = [min_idxs[i] for i in range(len(min_idxs)) if min_vals[i] < max_dist]
    res = np.unique(min_idxs_filtered)
    return res


def get_best_grasps(s_confidence, grasp_poses, n=10):
    n_min_idxs, _ = argmax_n(s_confidence, n)
    best_grasps = grasp_poses[n_min_idxs]
    return best_grasps


def get_gf_v_vec():
    gripper_width = 0.08
    left_y = -0.5 * gripper_width
    right_y = 0.5 * gripper_width
    mid_z = 0.062
    top_z = 0.112
    a = [0, 0, 0]
    b = [0, left_y, mid_z]
    c = [0, right_y, mid_z]
    d = [0, left_y, top_z]
    e = [0, right_y, top_z]
    gf_v = np.array([a, b, c, d, e]) - np.array([0, 0, 0.062])
    return gf_v


def v_vec_from_grasp(grasp_pose):
    # v in canonical frame
    gf_v = get_gf_v_vec()
    # v in object frame
    wf_v = np.array([grasp_pose @ np.append(x, 1) for x in gf_v])[:, :3]
    return wf_v


def generate_gt_v(xyz_points, successful_grasps):
    min_idxs, _ = knn(xyz_points, successful_grasps[:, :3, 3])
    closest_grasps = successful_grasps[min_idxs]
    v_vecs = [v_vec_from_grasp(grasp_pose) for grasp_pose in closest_grasps]
    return v_vecs
