import numpy as np
import numpy.typing as npt

from src.mathUtils import *
from src.reader import ReadIMU
from src.imu import IMU
from src.plot import Pose
from src.ekf import Initializer, EKF


class Tracker:
    def __init__(self, T: float):
        self.imu: IMU = IMU(T)
        self.ekf: EKF = EKF(T)
        self.pose: Pose = Pose(T)
        self.config = ReadIMU.get_config()

        self.ekf.set_init_values(self._initializer())

    def _initializer(self) -> list[float | npt.NDArray[np.float64]]:
        """Create initializer and take first N messages"""
        initial: Initializer = Initializer()
        init_values: list[float | npt.NDArray[np.float64]] | None = None

        while not initial.inited and init_values is None:
            try:
                imu_data = next(self.get_imu())
                init_values = initial.collect(imu_data)
            except ValueError:
                break

        if len(init_values) == 7:
            return init_values
        else:
            raise ValueError("Initialization is broken")

    def _data_prepare(
        self, imu_vector: npt.NDArray[np.float64]
    ) -> list[npt.NDArray[np.float64]]:
        """Prepare input data to EKF-readable format"""
        wt = imu_vector[np.newaxis, self.config["w"]].T
        at = imu_vector[np.newaxis, self.config["g"]].T
        mt = normalized(imu_vector[np.newaxis, self.config["g"]].T)

        return wt, at, mt

    def get_imu(self) -> npt.NDArray[np.float64]:
        """Response for new imu-data"""
        for q in self.imu:
            yield q

    def track(self):
        wt, at, mt = self._data_prepare(next(self.get_imu()))
        orin, pos = self.ekf.process(wt, at, mt)
        self.pose.append_data(orin, pos)
        return orin, pos
