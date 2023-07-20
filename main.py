from src.tracker import Tracker


class Runner:
    def __init__(self):
        T: float = 1 / 25
        self.tracker: Tracker = Tracker(T)

    def run(self):
        while True:
            R, pose = self.tracker.track()
            print(R, pose)


if __name__ == "__main__":
    imu_plot: Runner = Runner()
    imu_plot.run()
