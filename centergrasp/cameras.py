import numpy as np
import open3d as o3d
from dataclasses import dataclass
from simnet.lib.camera import convert_homopixels_to_pixels, convert_points_to_homopoints


@dataclass
class CameraParams:
    """
    Pinhole Camera Model
    """

    width: int
    height: int
    K: np.ndarray

    @property
    def fx(self) -> float:
        return self.K[0, 0]

    @property
    def fy(self) -> float:
        return self.K[1, 1]

    @property
    def cx(self) -> float:
        return self.K[0, 2]

    @property
    def cy(self) -> float:
        return self.K[1, 2]

    @property
    def fov_x_rad(self) -> float:
        return 2 * np.arctan2(self.width, 2 * self.fx)

    @property
    def fov_y_rad(self) -> float:
        return 2 * np.arctan2(self.height, 2 * self.fy)

    @property
    def fov_x_deg(self) -> float:
        return np.rad2deg(self.fov_x_rad)

    @property
    def fov_y_deg(self) -> float:
        return np.rad2deg(self.fov_y_rad)

    def to_open3d(self) -> o3d.camera.PinholeCameraIntrinsic:
        return o3d.camera.PinholeCameraIntrinsic(
            width=self.width,
            height=self.height,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
        )

    @classmethod
    def from_ideal_params(cls, width, height, f_xy):
        cx = width / 2
        cy = height / 2
        K = np.array([[f_xy, 0.0, cx], [0.0, f_xy, cy], [0.0, 0.0, 1.0]])
        return cls(width, height, K)

    def downsampled(self, factor: int):
        assert self.width % factor == 0, "width must be divisible by factor"
        assert self.height % factor == 0, "height must be divisible by factor"
        new_K = self.K.copy()
        new_K[:2, :] /= factor
        return CameraParams(
            width=self.width // factor,
            height=self.height // factor,
            K=new_K,
        )

    def cropped(self, crop_height: int = 0, crop_width: int = 0):
        K = self.K.copy()
        K[0, 2] -= crop_width / 2
        K[1, 2] -= crop_height / 2
        return CameraParams(
            width=self.width - crop_width,
            height=self.height - crop_height,
            K=K,
        )


def _normalize(vec):
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-9)


def look_at_z(eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0, 0, -1])):
    """
    Camera looks in positive z-direction
    """
    eye = np.array(eye)
    target = np.array(target)
    zaxis = _normalize(target - eye)
    xaxis = _normalize(np.cross(up, zaxis))
    yaxis = _normalize(np.cross(zaxis, xaxis))
    m = np.eye(4)
    m[:3, 0] = xaxis
    m[:3, 1] = yaxis
    m[:3, 2] = zaxis
    m[:3, 3] = eye
    return m


def look_at_x(eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0, 0, 1])):
    """
    Camera looks in positive x-direction
    """
    eye = np.array(eye)
    target = np.array(target)
    xaxis = _normalize(target - eye)
    yaxis = _normalize(np.cross(up, xaxis))
    zaxis = _normalize(np.cross(xaxis, yaxis))
    m = np.eye(4)
    m[:3, 0] = xaxis
    m[:3, 1] = yaxis
    m[:3, 2] = zaxis
    m[:3, 3] = eye
    return m


def sample_cam_pose_shell(center: np.ndarray, coi_half_size: float) -> np.ndarray:
    """
    :param center: Center of the scene
    :param coi_half_size: Half size of the cube around the center of interest (cube of interest)
    """
    point_of_interest = center + np.random.uniform([-coi_half_size] * 3, [coi_half_size] * 3)
    cam_position = sample_position_shell(
        center=point_of_interest,
        radius_min=0.3,
        radius_max=0.6,
        elevation_min=30,
        elevation_max=80,
    )
    cam_pose = look_at_x(eye=cam_position, target=point_of_interest)
    return cam_pose


def sample_cam_poses_shell(center: np.ndarray, coi_half_size: float, num_poses: int) -> np.ndarray:
    """
    :param center: Center of the scene
    :param coi_half_size: Half size of the cube around the center of interest (cube of interest)
    :param num_poses: Number of poses to sample
    """
    poses_out = np.array([sample_cam_pose_shell(center, coi_half_size) for _ in range(num_poses)])
    return poses_out


def sample_position_shell(
    center: np.ndarray,
    radius_min: float,
    radius_max: float,
    elevation_min: float = -90,
    elevation_max: float = 90,
    azimuth_min: float = -180,
    azimuth_max: float = 180,
) -> np.ndarray:
    """
    Samples a point from the volume between two spheres (radius_min, radius_max). Optionally the
    spheres can be constraint by setting elevation and azimuth angles. E.g. if you only want to
    sample in the upper hemisphere set elevation_min = 0. Instead of sampling the angles and radius
    uniformly, sample the shell volume uniformly. As a result, there will be more samples at larger
    radii.

    :param center: Center shared by both spheres.
    :param radius_min: Radius of the smaller sphere.
    :param radius_max: Radius of the bigger sphere.
    :param elevation_min: Minimum angle of elevation in degrees. Range: [-90, 90].
    :param elevation_max: Maximum angle of elevation in degrees. Range: [-90, 90].
    :param azimuth_min: Minimum angle of azimuth in degrees. Range: [-180, 180].
    :param azimuth_max: Maximum angle of azimuth in degrees. Range: [-180, 180].
    :return: A sampled point.
    """
    assert -180 <= azimuth_min <= 180, "azimuth_min must be in range [-180, 180]"
    assert -180 <= azimuth_max <= 180, "azimuth_max must be in range [-180, 180]"
    assert -90 <= elevation_min <= 90, "elevation_min must be in range [-90, 90]"
    assert -90 <= elevation_min <= 90, "elevation_max must be in range [-90, 90]"
    assert azimuth_min < azimuth_max, "azimuth_min must be smaller than azimuth_max"
    assert elevation_min < elevation_max, "elevation_min must be smaller than elevation_max"

    radius = radius_min + (radius_max - radius_min) * np.cbrt(np.random.rand())

    # rejection sampling
    constr_fulfilled = False
    while not constr_fulfilled:
        direction_vector = np.random.randn(3)
        direction_vector /= np.linalg.norm(direction_vector)

        # https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
        xy = direction_vector[0] * direction_vector[0] + direction_vector[1] * direction_vector[1]
        elevation = np.arctan2(direction_vector[2], np.sqrt(xy))
        azimuth = np.arctan2(direction_vector[1], direction_vector[0])

        elev_constraint = np.deg2rad(elevation_min) < elevation < np.deg2rad(elevation_max)
        azim_constraint = np.deg2rad(azimuth_min) < azimuth < np.deg2rad(azimuth_max)
        constr_fulfilled = elev_constraint and azim_constraint

    # Get the coordinates of a sampled point inside the shell
    position = direction_vector * radius + center

    return position


def project(points_3d: np.ndarray, camera_intrinsics: np.ndarray):
    """
    Project a set of points in the camera frame onto the image plane

        points_3d: 3xN or 4xN

    return 2xN
    """
    if points_3d.shape[0] == 3:
        points_homo = convert_points_to_homopoints(points_3d)
    else:
        assert points_3d.shape[0] == 4
        points_homo = points_3d
    if camera_intrinsics.shape == (3, 3):
        camera_intrinsics = np.concatenate((camera_intrinsics, np.zeros((3, 1))), axis=1)
    points_image = camera_intrinsics @ points_homo
    return convert_homopixels_to_pixels(points_image)


@dataclass
class CameraConventions:
    """
    Different libraries use different camera coordinate conventions:
    - OpenGL/Blender: +X is right, +Y is up, and +Z is pointing back
    - OpenCV/COLMAP: +X is right, +Y is down, and +Z is pointing forward
    - Robotics/SAPIEN: +X is forward, +Y is left, +Z is up
    """

    opengl_T_opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]).T
    opengl_T_robotics = np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]).T
    opencv_T_robotics = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]).T
    opencv_T_opengl = opengl_T_opencv.T
    robotics_T_opengl = opengl_T_robotics.T
    robotics_T_opencv = opencv_T_robotics.T
    opengl_R_opencv = opengl_T_opencv[:3, :3]
    opengl_R_robotics = opengl_T_robotics[:3, :3]
    opencv_R_robotics = opencv_T_robotics[:3, :3]
    opencv_R_opengl = opencv_T_opengl[:3, :3]
    robotics_R_opencv = robotics_T_opencv[:3, :3]
    robotics_R_opengl = robotics_T_opengl[:3, :3]
