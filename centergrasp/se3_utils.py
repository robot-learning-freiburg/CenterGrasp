import numpy as np
import spatialmath as sm


def translate_pose_y(pose, transl):
    rel_t = sm.SE3.Ty(transl)
    wTgoal = pose * rel_t
    return wTgoal


def rotate_pose_y(pose, rot_deg):
    rel_transf = sm.SE3.Ry(rot_deg, "deg")
    wTgoal = pose * rel_transf
    return wTgoal


def align_x_to_normal(point, normal):
    rot_x = normal / np.linalg.norm(normal)  # make sure it is actually normal
    y = np.array([0.0, 0.0, 1.0])  # pick some direction
    if 1 - np.abs(np.dot(y, rot_x)) < 1e-3:
        y = np.array([0.0, 1.0, 0.0])  # pick a different direction
    rot_y = y - np.dot(y, rot_x) * rot_x  # find projection
    rot_y /= np.linalg.norm(rot_y)  # normalize it
    rot_z = np.cross(rot_x, rot_y)  # find last vector
    rot = sm.SO3(np.vstack((rot_x, rot_y, rot_z)).T)
    pose = sm.SE3.Rt(R=rot, t=point)
    return pose


def align_y_to_normal(point, normal):
    rot_y = normal / np.linalg.norm(normal)  # make sure it is actually normal
    z = np.array([0.0, 0.0, 1.0])  # pick some direction
    if 1 - np.abs(np.dot(z, rot_y)) < 1e-3:
        z = np.array([0.0, 1.0, 0.0])  # pick a different direction
    rot_z = z - np.dot(z, rot_y) * rot_y  # find projection
    rot_z /= np.linalg.norm(rot_z)  # normalize it
    rot_x = np.cross(rot_y, rot_z)  # find last vector
    rot = sm.SO3(np.vstack((rot_x, rot_y, rot_z)).T)
    pose = sm.SE3.Rt(R=rot, t=point)
    return pose


def align_z_to_normal(point, normal):
    rot_z = normal / np.linalg.norm(normal)  # make sure it is actually normal
    x = np.array([1.0, 0.0, 0.0])  # pick some direction
    if 1 - np.abs(np.dot(x, rot_z)) < 1e-3:
        x = np.array([0.0, 1.0, 0.0])  # pick a different direction
    rot_x = x - np.dot(x, rot_z) * rot_z  # find projection
    rot_x /= np.linalg.norm(rot_x)  # normalize it
    rot_y = np.cross(rot_z, rot_x)  # find last vector
    rot = sm.SO3(np.vstack((rot_x, rot_y, rot_z)).T)
    pose = sm.SE3.Rt(R=rot, t=point)
    return pose


def pose_4x4_to_flat(pose_4x4: np.ndarray) -> np.ndarray:
    rot = pose_4x4[:3, :3]
    trans = pose_4x4[:3, 3]
    pose_flat = np.concatenate([rot.flatten(), trans])
    return pose_flat


def pose_flat_to_4x4(pose_flat: np.ndarray) -> np.ndarray:
    rot = pose_flat[:9].reshape((3, 3))
    trans = pose_flat[9:]
    pose_4x4 = np.eye(4)
    pose_4x4[:3, :3] = rot
    pose_4x4[:3, 3] = trans
    return pose_4x4
