import os
import asyncio
import time
from io import TextIOWrapper
from typing import Optional

import numpy as np
import numpy.typing as npt

from src.mathUtils import AxisOrder


class AsyncIMU:
    def __init__(self, path: str, filename: str):
        self.reader = ReadIMU(path, filename)
        self.loop = asyncio.get_event_loop()

    def get_data(self) -> npt.NDArray[np.float64]:
        return self.loop.run_until_complete(self.__async__get())

    async def __async__get(self):
        async with self.reader as imu:
            return await imu.get()


class ReadIMU:
    """
    Read IMU file.
    Data format:
    [q0 .. q3, g0 .. g2, w0 .. w2, m0 .. m2]
    """

    def __init__(self, path: str, filename: str, T: float = 1 / 25):
        self.file: str = os.path.join(path, filename)
        self.T: float = T
        self._config: dict[str, slice] = self.get_config()
        self._file_r: TextIOWrapper = open(self.file, "r")
        self._data: Optional[npt.NDArray[np.float64]] = None

    def _read_imu(self) -> npt.NDArray[np.float64]:
        time.sleep(self.T)
        line: list[str] = self._file_r.readline().split(" ")
        self._data = np.array(line).astype(float)

    @staticmethod
    def get_config() -> dict[str, slice]:
        config = {
            "q": slice(0, 4),
            "g": slice(4, 7),
            "w": slice(7, 10),
            "m": slice(10, 13),
        }
        return config

    def update(self) -> npt.NDArray[np.float64]:
        self._read_imu()

    def data(self, val: Optional[str] = None) -> tuple[npt.NDArray[np.float64]]:
        match val:
            case "q":
                return self._data[self._config["q"]]
            case "g":
                return AxisOrder(self._data[self._config["g"]])
            case "w":
                return AxisOrder(self._data[self._config["w"]])
            case "m":
                return AxisOrder(self._data[self._config["m"]])
            case _:
                return np.hstack(self.data(x) for x in self.get_config().keys())
