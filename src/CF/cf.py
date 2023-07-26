import numpy as np
import numpy.typing as npt

from src.mathUtils import *

class ComplemetaryFilter:
    def __init__(self, T: float):
        self.dt = T
        self.epoch: int = 1
        self.alpha: float = 0.65

    def set_init_values(self, values: list[float | npt.NDArray[np.float64]]):
        (
            self.q_, 
            self.roll_, 
            self.pitch_, 
            self.yaw_
        ) = values

    def process(self,
                qt: npt.NDArray[np.float64],
                wt: npt.NDArray[np.float64],
                at: npt.NDArray[np.float64],
                mt: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float]]:

        T = np.array([0, 0, 0])[:, np.newaxis]
        return [np.hstack([QuaternionToRotationMatrix(r), T]) for r in self.SlerpCalc(qt,wt,at,mt)]


    def EulerCalc(self,
                  wt: npt.NDArray[np.float64],
                  at: npt.NDArray[np.float64],
                  mt: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Current Euler from IMU"""
        roll_g = rollAcc(at)
        pitch_g = pitchAcc(at)
        roll_w = rollGyro(self.roll_, wt, self.dt)
        pitch_w = pitchGyro(self.pitch_, wt, self.dt)

        roll = ComplementaryFilter(roll_w, roll_g, self.alpha)
        pitch = ComplementaryFilter(pitch_w, pitch_g, self.alpha)
        yaw = yawMag(roll, pitch, mt)

        return np.array([roll, pitch, yaw])


    def SlerpCalc(self,
                  qt: npt.NDArray[np.float64],
                  wt: npt.NDArray[np.float64],
                  at: npt.NDArray[np.float64],
                  mt: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Returns Euler-angles after Slerp"""
        q1 = QuaternionMotionAppend(self.q_, wt, self.dt)
        q2 = EulerToQuaternion(self.EulerCalc(wt, at, mt)) # ? Experiment with combos
        q3 = qt

        self.roll_, self._pitch, self._yaw = QuaternionToEuler(q3)
        self.q_ = q3

        # slerp = Slerp(q1, q2, self.alpha) # is not working
        # return 
        # return np.array([QuaternionToEuler(q) for q in [q1, q2, q3]])

        return np.array([q1, q2, q3])