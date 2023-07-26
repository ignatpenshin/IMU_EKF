from src.EKF.tracker import Tracker as EKFTracker
from src.CF.tracker import Tracker as CFTracker


class Runner:
    def __init__(self):
        T: float = 1 / 25
        self.tracker_1: EKFTracker = EKFTracker(T)
        #self.tracker_2: CFTracker = CFTracker(T)

    def run(self):
        while True:
            R, pose = self.tracker_1.track()
            #R, pose = self.tracker_2.track()


if __name__ == "__main__":
    imu_plot: Runner = Runner()
    imu_plot.run()
