import numpy as np
import numpy.typing as npt

from src.reader import ReadIMU
from src.config import Vars
from src.mathUtils import *


class IMU:
    def __init__(self, T: float):
        self.imu: ReadIMU = ReadIMU(Vars.PATH, Vars.FILE, T)

    def __iter__(self):
        return self

    def __next__(self) -> npt.NDArray[np.float64]:
        self.imu.update()
        return self.imu.data()
