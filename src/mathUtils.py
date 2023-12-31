from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def rollAcc(acc: npt.NDArray[np.float64]) -> float:
    return np.rad2deg(np.arctan(acc[1] / np.hypot(acc[0], acc[2])))


def pitchAcc(acc: npt.NDArray[np.float64]) -> float:
    return np.rad2deg(np.arctan(-acc[0] / np.hypot(acc[1], acc[2])))


def rollGyro(rollPrev: float, gyro: npt.NDArray[np.float64], dt: float) -> float:
    return rollPrev + gyro[0] * dt


def pitchGyro(pitchPrev: float, gyro: npt.NDArray[np.float64], dt: float) -> float:
    return pitchPrev + gyro[1] * dt


def ComplementaryFilter(x: float, y: float, alpha: float) -> float:
    return alpha * x + (1 - alpha) * y


def yawMag(roll: float, pitch: float, mag: npt.NDArray[np.float64]) -> float | None:
    magX: np.float = mag[2] * np.sin(roll) - mag[1] * np.cos(roll)
    magY: np.float = (
        mag[0] * np.cos(roll)
        + mag[1] * np.sin(pitch) * np.sin(roll)
        + mag[2] * np.sin(pitch) * np.sin(roll)
    )
    return np.rad2deg(np.arctan2(magY, magX))


def EulerToQuaternion(euler: list[float]) -> npt.NDArray[np.float64]:
    r: R = R.from_euler("xyz", euler, degrees=True)
    return r.as_quat()


def QuaternionToEuler(quat: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    r: R = R.from_quat(quat)
    return r.as_euler("xyz", degrees=True)


def QuaternionMotionDelta(
    quat: npt.NDArray[np.float64], gyro: npt.NDArray[np.float64], dt: float
) -> npt.NDArray[np.float64]:
    dQ = np.array(
        [
            [-quat[1], -quat[2], -quat[3]],
            [quat[0], -quat[3], quat[2]],
            [quat[3], quat[0], -quat[1]],
            [-quat[2], quat[1], quat[0]],
        ]
    )

    return 0.5 * dt * dQ @ gyro.T


def QuaternionMotionAppend(
    quat_: npt.NDArray[np.float64], gyro: npt.NDArray[np.float64], dt: float
) -> npt.NDArray[np.float64]:
    q_ = quat_ + QuaternionMotionDelta(quat_, gyro, dt)
    return q_ / np.linalg.norm(q_, 2)


def QuaternionToRotationMatrix(
    quat: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    r: R = R.from_quat(quat)
    return r.as_matrix()


def Slerp(
    q1: npt.NDArray[np.float64], q2: npt.NDArray[np.float64], p: float
) -> npt.NDArray[np.float64]:
    rotations = np.array([R.from_quat([q1]), R.from_quat([q2])])
    times = np.array([0, 1])
    slerp = Slerp(times, rotations)

    return slerp(
        np.array([p,])
    ).as_euler("xyz", degrees=True)


def normalized(x: Any) -> Any:
    try:
        return x / np.linalg.norm(x)
    except:
        return x


def I(n: int) -> npt.NDArray[np.float64]:
    return np.eye(n)


def F(wt: npt.NDArray[np.float64], dt: float) -> npt.NDArray[np.float64]:
    """State Transfer matrix"""
    w = wt.T[0]
    Omega = np.array(
        [
            [0, -w[0], -w[1], -w[2]],
            [w[0], 0, w[2], -w[1]],
            [w[1], -w[2], 0, w[0]],
            [w[2], w[1], -w[0], 0],
        ]
    )
    return I(4) + 0.5 * dt * Omega


def G(q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    q = q.T[0]
    return 0.5 * np.array(
        [
            [-q[1], -q[2], -q[3]],
            [q[0], -q[3], q[2]],
            [q[3], q[0], -q[1]],
            [-q[2], q[1], q[0]],
        ]
    )


def skew(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    takes in a 3d column vector
    returns its Skew-symmetric matrix
    """

    x = x.T[0]
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def rotate(q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    rotation transformation matrix
    nav frame to body frame as q is expected to be q^nb
    R(q) @ x to rotate x
    """
    qv = q[1:4, :]
    qc = q[0]
    return (qc**2 - qv.T @ qv) * I(3) - 2 * qc * skew(qv) + 2 * qv @ qv.T


def Hhelper(
    q: npt.NDArray[np.float64], vector: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    # just for convenience
    x = vector.T[0][0]
    y = vector.T[0][1]
    z = vector.T[0][2]
    q0 = q.T[0][0]
    q1 = q.T[0][1]
    q2 = q.T[0][2]
    q3 = q.T[0][3]

    h = np.array(
        [
            [
                q0 * x - q3 * y + q2 * z,
                q1 * x + q2 * y + q3 * z,
                -q2 * x + q1 * y + q0 * z,
                -q3 * x - q0 * y + q1 * z,
            ],
            [
                q3 * x + q0 * y - q1 * z,
                q2 * x - q1 * y - q0 * z,
                q1 * x + q2 * y + q3 * z,
                q0 * x - q3 * y + q2 * z,
            ],
            [
                -q2 * x + q1 * y + q0 * z,
                q3 * x + q0 * y - q1 * z,
                -q0 * x + q3 * y - q2 * z,
                q1 * x + q2 * y + q3 * z,
            ],
        ]
    )
    return 2 * h


def H(
    q: npt.NDArray[np.float64], gn: npt.NDArray[np.float64], mn: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Measurement matrix
    """

    H1 = Hhelper(q, gn)
    H2 = Hhelper(q, mn)

    return np.vstack((-H1, H2))


def AxisOrder(data: npt.NDArray[np.float64]):
    R = np.array([[0, 0, 1], 
                  [-1, 0, 0], 
                  [0, -1, 0]])
    
    return R @ data.T
