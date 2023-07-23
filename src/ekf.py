import numpy as np
import numpy.typing as npt

from src.mathUtils import *
from src.reader import ReadIMU


class Initializer:
    """
    Static initialization of values used by EKF algorithm:
    (gn, g0, mn, gyro_noise, gyro_bias, acc_noise, mag_noise)
    """

    def __init__(self, size: int = 30):
        self.inited: bool = False
        self._size: int = size
        self._data: list[npt.NDArray[np.float64] | None] = []
        self.noise_coefficient: dict[str, float] = {"w": 100.0, "a": 100.0, "m": 10.0}

    def collect(
        self, imu_vector: npt.NDArray[np.float64]
    ) -> tuple[float | npt.NDArray[np.float64]] | None:
        if not self.inited:
            self._data.append(imu_vector)
            if len(self._data) == self._size:
                self._data = np.vstack(self._data)
                self.inited = True
                return self._calculate_init_values()
        else:
            raise ValueError("Already initialized")

    def _calculate_init_values(self) -> tuple[float | npt.NDArray[np.float64]]:
        config = ReadIMU.get_config()

        w = self._data[:, config["w"]]
        a = self._data[:, config["g"]]
        m = self._data[:, config["m"]]

        gn, g0 = self.gravitiy_initial(a)
        mn = self.magnetic_initial(m)
        avar, wvar, mvar = self.noise_covar_init(a, w, m)

        gyro_noise, gyro_bias, acc_noise, mag_noise = self.sensor_noise(
            avar, wvar, mvar, w
        )

        return (gn, g0, mn, gyro_noise, gyro_bias, acc_noise, mag_noise)

    @staticmethod
    def gravitiy_initial(
        a: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], float]:
        """
        initial gravity vector gn,
        initial magnitude of gn: g0
        """
        gn = -a.mean(axis=0)
        gn = gn[:, np.newaxis]
        g0 = np.linalg.norm(gn)

        return gn, g0

    @staticmethod
    def magnetic_initial(m: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """magnetic field: mn"""
        mn = m.mean(axis=0)
        mn = normalized(mn)[:, np.newaxis]
        return mn

    @staticmethod
    def noise_covar_init(
        a: npt.NDArray[np.float64],
        w: npt.NDArray[np.float64],
        m: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64]]:
        avar = a.var(axis=0)
        wvar = w.var(axis=0)
        mvar = m.var(axis=0)

        print("acc var: %s, norm: %s" % (avar, np.linalg.norm(avar)))
        print("ang var: %s, norm: %s" % (wvar, np.linalg.norm(wvar)))
        print("mag var: %s, norm: %s" % (mvar, np.linalg.norm(mvar)))

        return avar, wvar, mvar

    def sensor_noise(self, avar, wvar, mvar, w) -> tuple[float]:
        """
        Define sensor noise with noise_coefficient,
        defined manually + gyro beas via gyro-data.mean()
        """

        gyro_noise = self.noise_coefficient["w"] * np.linalg.norm(wvar)
        gyro_bias = w.mean(axis=0)
        acc_noise = self.noise_coefficient["a"] * np.linalg.norm(avar)
        mag_noise = self.noise_coefficient["m"] * np.linalg.norm(mvar)

        return (gyro_noise, gyro_bias, acc_noise, mag_noise)


class EKF:
    def __init__(self, T: float):
        self.dt = T
        self.P = 1e-10 * I(4)
        self.q = np.array([[1, 0, 0, 0]]).T
        self.init_ori = I(3)

        self.v = np.zeros((3, 1))
        self.p = np.array([[0, 0, 0]]).T

    def set_init_values(self, values: list[float | npt.NDArray[np.float64]]):
        (
            self.gn,
            self.g0,
            self.mn,
            self.gyro_noise,
            self.gyro_bias,
            self.acc_noise,
            self.mag_noise,
        ) = values

    def process(
        self,
        wt: npt.NDArray[np.float64],
        at: npt.NDArray[np.float64],
        mt: npt.NDArray[np.float64],
    ):
        self.predict(wt, at, mt)
        self.update(at, mt)
        self.post_correction()

        return self.navigation_frame(at)

    def predict(
        self,
        wt: npt.NDArray[np.float64],
        at: npt.NDArray[np.float64],
        mt: npt.NDArray[np.float64],
    ) -> None:
        Ft = F(wt, self.dt)
        Gt = G(self.q)
        Q = (self.gyro_noise * self.dt) ** 2 * Gt @ Gt.T

        self.q = normalized(Ft @ self.q)
        self.P = Ft @ self.P @ Ft.T + Q

    def update(self, at: npt.NDArray[np.float64], mt: npt.NDArray[np.float64]) -> None:
        # Acc & Mag normailized prediction
        pa = normalized(-rotate(self.q) @ self.gn)
        pm = normalized(rotate(self.q) @ self.mn)

        # Residual
        Eps = np.vstack((normalized(at), mt)) - np.vstack((pa, pm))

        # R = internal error + external error (Sensor noise)
        Ra = [
            (self.acc_noise / np.linalg.norm(at)) ** 2
            + (1 - self.g0 / np.linalg.norm(at)) ** 2
        ] * 3
        Rm = [self.mag_noise**2] * 3
        R = np.diag(Ra + Rm)

        # Kalman Gain
        Ht = H(self.q, self.gn, self.mn)
        S = Ht @ self.P @ Ht.T + R
        K = self.P @ Ht.T @ np.linalg.inv(S)

        # Update
        self.q = self.q + K @ Eps
        self.P = self.P - K @ Ht @ self.P

    def post_correction(self):
        # make sure P is symmertical
        self.q = normalized(self.q)
        self.P = 0.5 * (self.P + self.P.T)

    def navigation_frame(
        self, at: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64]]:
        # Acceleration on nav-frame
        conj = -I(4)
        conj[0, 0] = 1
        an = rotate(conj @ self.q) @ at + self.gn

        # Orientation on nav-frame
        orin = rotate(conj @ self.q) @ self.init_ori

        # Velocity & Pose
        self.v = self.v + an * self.dt
        self.p = self.p + self.v * self.dt + 0.5 * an * self.dt**2

        print("-----------------")
        print("Gn: ", self.gn)
        print("At: ", at)
        print("An: ", an)
        print("V: ", self.v)
        print("P: ", self.p)
        print("-----------------")

        return orin, self.p
