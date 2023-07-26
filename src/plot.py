import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d, proj3d


class Pose:
    def __init__(self, T: float = 1 / 25):
        self.pose = []
        self.orix = []
        self.oriy = []
        self.oriz = []
        self.plotter: Plotter = Plotter(T)

    def append_data(
        self, orin: npt.NDArray[np.float64], pose: npt.NDArray[np.float64]
    ) -> None:
        self.pose.append(pose.T[0])
        self.orix.append(orin.T[0, :])
        self.oriy.append(orin.T[1, :])
        self.oriz.append(orin.T[2, :])
        self.plotter.plot_orientation(orin, pose, history=np.vstack(self.pose))


class Plotter:
    def __init__(self, T: float = 1 / 25):
        self.T = T
        plt.ion()
        self.fig = plt.figure(figsize=(1.5, 1.5))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.dims = np.array([0.8, 0.4, 0.1])

    def plot_axes(self, axis_points, T):
        """Plots the local axes on the plot"""
        for i in range(3):
            end = axis_points[:, i]
            x = [0, end[0]]
            y = [0, end[1]]
            z = [0, end[2]]
            self.ax.plot3D(x + T[0], y + T[1], z + T[2])

    def plot_orientation(
        self, R: npt.NDArray[np.float64], T: npt.NDArray[np.float64], history
    ) -> None:
        """Plot the IMU given an orientation"""
        normals = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        ## Get local axis coordinates and phone cuboid
        coords = R @ normals

        self.ax.clear()
        # self.ax.add_collection3d(pc)
        # self.ax.set_xlim3d(-50, 50) #1.5
        # self.ax.set_ylim3d(-50, 50)
        # self.ax.set_zlim3d(-50, 50)
        self.plot_axes(coords, T)
        self.ax.plot(history[-300:, 0], history[-300:, 1], history[-300:, 2], "o")

        plt.pause(self.T)

    def R_from_q(self, q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        orientation = np.deg2rad(q)

        ca, cb, cg = np.cos(orientation)
        sa, sb, sg = np.sin(orientation)

        # Correct for sign conventions
        sa = -sa
        sg = -sg

        # Get rotation matrix
        R = np.array(
            [
                [ca * cb, ca * sb * sg - sa * cg, ca * sb * cg + sa * sg],
                [sa * cb, sa * sb * sg + ca * cg, sa * sb * cg - ca * sg],
                [-sb, cb * sg, cb * cg],
            ]
        )

        return R
