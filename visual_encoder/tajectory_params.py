from visual_encoder.displacement_params import DisplacementParams
from visual_encoder.trajectory_estimators import compute_total_trajectory
from visual_encoder.phase_correlation import crosspower_spectrum
from visual_encoder.svd_decomposition import phase_unwrapping, linear_regression
from scipy.sparse.linalg import svds
import numpy as np

# This class contains all parameters required in the global displacement estimation algorithm

class TrajectoryParams(DisplacementParams):
    def __init__(self, disp_params: DisplacementParams, x0=0, y0=0):
        super().__init__(method=disp_params.method, spatial_window=disp_params.spatial_window,
                         frequency_window=disp_params.frequency_window)
        self.x0 = x0
        self.y0 = y0
        self.coords = None
        self.n_shots = None

    def get_init_coord(self):
        return self.x0, self.y0

    def get_coords(self, shot=None):
        if shot is not None:
            return self.coords[shot, :]
        else:
            return self.coords

    def compute_cps(self, img_list, shot):
        f = img_list[shot]
        g = img_list[shot - 1]
        q, Q = crosspower_spectrum(f, g, self.frequency_window)
        return q, Q

    def compute_svd(self, img_list, shot):
        f = img_list[shot]
        g = img_list[shot - 1]
        q, Q = crosspower_spectrum(f, g, method=self.frequency_window)
        qu, s, qv = svds(Q, k=1)
        return qu, s, qv

    def estimate_shift(self, pu_unwrapped, N):
        r = np.arange(0, pu_unwrapped.size)
        M = r.size // 2
        # Choosing a smaller window:
        x = r[M - 50:M + 50]
        y = pu_unwrapped[M - 50:M + 50]
        mu, c = linear_regression(x, y)
        delta = -mu * N / (2 * np.pi)
        return delta, mu, c, x, y

    def set_coords(self, new_coords):
        self.coords = new_coords

    def compute_total_trajectory(self, img_list):
        self.coords = compute_total_trajectory(img_list, self)
