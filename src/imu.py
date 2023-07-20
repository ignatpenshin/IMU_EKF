import numpy as np
import numpy.typing as npt

from src.reader import ReadIMU
from src.config import Vars
from src.mathUtils import CalcUtils


class IMU:
    def __init__(self, T: float):
        self.imu: ReadIMU = ReadIMU(Vars.PATH, Vars.FILE, T)
        self.epoch: int = 1
        self.alpha: float = 0.65
        self.initValues()

    def __iter__(self):
        return self

    def __next__(self) -> npt.NDArray[np.float64]:
        self.imu.update()
        return self.imu.data()  # self.SlerpCalc()

    def initValues(self):
        """
        q0, roll0, pitch0, yaw0 init
        pitch, yaw, roll --> X, Y, Z rotations
        """
        self.imu.update()
        self.q_ = self.imu.data("q")
        self.roll_ = CalcUtils.rollAcc(self.imu.data("g"))
        self.pitch_ = CalcUtils.pitchAcc(self.imu.data("g"))
        self.yaw_ = CalcUtils.yawMag(self.roll_, self.pitch_, self.imu.data("m"))

    def EulerCalc(self) -> npt.NDArray[np.float64]:
        """Current Euler from IMU"""
        roll_g = CalcUtils.rollAcc(self.imu.data("g"))
        pitch_g = CalcUtils.pitchAcc(self.imu.data("g"))
        roll_w = CalcUtils.rollGyro(self.roll_, self.imu.data("w"), self.imu.T)
        pitch_w = CalcUtils.pitchGyro(self.pitch_, self.imu.data("w"), self.imu.T)

        roll = CalcUtils.ComplementaryFilter(roll_w, roll_g, self.alpha)
        pitch = CalcUtils.ComplementaryFilter(pitch_w, pitch_g, self.alpha)
        yaw = CalcUtils.yawMag(roll, pitch, self.imu.data("m"))

        return np.array([roll, pitch, yaw])

    def SlerpCalc(self) -> npt.NDArray[np.float64]:
        """Returns Euler-angles after Slerp"""
        # q1 = CalcUtils.QuaternionMotionAppend(self.q_, self.imu.data("w"), self.imu.T)
        # q2 = CalcUtils.EulerToQuaternion(self.EulerCalc()) # ? Experiment with combos
        q3 = self.imu.data("q")

        self.roll_, self._pitch, self._yaw = CalcUtils.QuaternionToEuler(q3)
        self.q_ = q3

        # slerp = CalcUtils.Slerp(q1, q2, self.alpha) # is not working
        # return np.array([q1, q2, q3])

        # return np.array([CalcUtils.QuaternionToEuler(q) for q in [q1, q2, q3]])
        return CalcUtils.QuaternionToEuler(q3)
