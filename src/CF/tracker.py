from src.plot import Pose
from src.CF.cf import ComplemetaryFilter
from src.imu import IMU
from src.mathUtils import *
from src.reader import ReadIMU


class Tracker:
    def __init__(self, T: float):
        self.imu: IMU = IMU(T)
        self.cf: ComplemetaryFilter = ComplemetaryFilter(T)
        self.pose: list[Pose] = [Pose(T) for i in range(3)]
        self.config = ReadIMU.get_config()

        self.cf.set_init_values(self._initializer())

    def _initializer(self) -> list[float | npt.NDArray[np.float64]]:
        """
        q0, roll0, pitch0, yaw0 init
        pitch, yaw, roll --> X, Y, Z rotations
        """
        imu_data = next(self.get_imu())
        q = imu_data[self.config["q"]]
        roll = rollAcc(imu_data[self.config["g"]])
        pitch = pitchAcc(imu_data[self.config["g"]])
        yaw = yawMag(roll, pitch, imu_data[self.config["m"]])

        return q, roll, pitch, yaw
    
    def _data_prepare(
        self, imu_vector: npt.NDArray[np.float64]
    ) -> list[npt.NDArray[np.float64]]:
        """Prepare input data to EKF-readable format"""
        qt = imu_vector[self.config["q"]]
        wt = imu_vector[self.config["w"]]
        at = imu_vector[self.config["g"]]
        mt = imu_vector[self.config["m"]]

        return qt, wt, at, mt

    def get_imu(self) -> npt.NDArray[np.float64]:
        """Response for new imu-data"""
        for q in self.imu:
            yield q

    def track(self):
        qt, wt, at, mt = self._data_prepare(next(self.get_imu()))
        r_pos_list = self.cf.process(qt, wt, at, mt)
        for i in range(len(r_pos_list)):
            pose = r_pos_list[i][:, 3][np.newaxis, :].T
            orin = r_pos_list[i][:, :3]
            self.pose[i].append_data(orin, pose)
        #self.pose.append_data(orin, pos)
        return orin, pose

        