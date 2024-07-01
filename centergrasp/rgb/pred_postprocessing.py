import copy
import numpy as np
import open3d as o3d
import spatialmath as sm
import mplib.pymp.fcl as fcl
from typing import List, Tuple
from centergrasp.cameras import CameraParams
from centergrasp.rgb.data_structures import RgbdDataNp
from centergrasp.sapien.stick_gripper_fcl import StickGripperFCL

# from centergrasp.sapien.franka_gripper_fcl import FrankaGripperFCL
from centergrasp.rgb.data_structures import FullObjPred, PostprObjPred

GRIPPER_FCL = StickGripperFCL()
# GRIPPER_FCL = FrankaGripperFCL()
# FRANKA_KIN = FrankaKinematics()


def collision_obj_from_pcd(pcd: np.ndarray):
    octree = fcl.OcTree(vertices=pcd, resolution=0.0001)
    return fcl.CollisionObject(octree, np.zeros(3), np.array([1, 0, 0, 0]))


def get_full_pcd(
    rgb_data: RgbdDataNp,
    cam_params: CameraParams,
    confidence_map: np.ndarray = None,
    project_valid_depth_only=False,
) -> o3d.geometry.PointCloud:
    o3d_camera_intrinsic = cam_params.to_open3d()
    if confidence_map is not None:
        depth = np.where(confidence_map < 20, rgb_data.depth, 0.0)
    else:
        depth = rgb_data.depth
    rgb_o3d = o3d.geometry.Image(rgb_data.rgb)
    depth_o3d = o3d.geometry.Image((depth * 1000).astype(np.uint16))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, depth_o3d, convert_rgb_to_intensity=False
    )
    full_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d_camera_intrinsic,
        project_valid_depth_only=project_valid_depth_only,
    )
    return full_pcd


def postprocess_predictions(
    rgb_data: RgbdDataNp,
    obj_preds: List[FullObjPred],
    num_grasps: int = 1,
    use_icp: bool = True,
    confidence_map: np.ndarray = None,
    joint_state: np.ndarray = None,
    wTbase: sm.SE3 = sm.SE3(),
    wTcam: sm.SE3 = sm.SE3(),
    camera_params: CameraParams = None,
) -> Tuple[List[PostprObjPred], o3d.geometry.PointCloud]:
    full_pcd = get_full_pcd(rgb_data, camera_params, confidence_map)
    full_pcd_fcl = collision_obj_from_pcd(np.asarray(full_pcd.points))
    # TODO: parallelize this loop
    postpr_predictions = [
        postprocess_prediction(
            obj_pred,
            full_pcd,
            full_pcd_fcl,
            num_grasps,
            use_icp,
            joint_state,
            wTbase,
            wTcam,
        )
        for obj_pred in obj_preds
    ]
    return postpr_predictions, full_pcd


def postprocess_prediction(
    obj_pred: FullObjPred,
    full_pcd: o3d.geometry.PointCloud,
    full_pcd_fcl: fcl.CollisionObject,
    num_grasps: int,
    use_icp: bool,
    joint_state: np.ndarray = None,
    wTbase: sm.SE3 = sm.SE3(),
    wTcam: sm.SE3 = sm.SE3(),
) -> PostprObjPred:
    # Shape postprocessing
    if use_icp:
        postpr_obj_pred = postprocess_shape(obj_pred, full_pcd)
    else:
        postpr_obj_pred = PostprObjPred(obj_pred.pc_o3d, obj_pred.grasp_poses)
    # Grasp postprocessing
    postpr_obj_pred = postprocess_grasps(
        postpr_obj_pred, full_pcd_fcl, num_grasps, joint_state, wTbase, wTcam
    )
    return postpr_obj_pred


def postprocess_shape(pred: FullObjPred, full_pcd: o3d.geometry.PointCloud) -> PostprObjPred:
    masked_pcd = full_pcd.select_by_index(np.flatnonzero(pred.bmask))
    masked_pcd.remove_non_finite_points()

    # Hidden point removal
    _, pt_map = pred.pc_o3d.hidden_point_removal([0.0, 0.0, 0.0], radius=100)
    pred_view_pcd = pred.pc_o3d.select_by_index(pt_map)

    # ICP
    masked_pcd.estimate_normals()
    masked_pcd.orient_normals_towards_camera_location()
    initial_shift = np.eye(4)
    initial_shift[:3, 3] = masked_pcd.get_center() - pred_view_pcd.get_center()
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source=pred_view_pcd,
        target=masked_pcd,
        max_correspondence_distance=0.05,
        init=initial_shift,
    )
    corrected_shape = copy.deepcopy(pred.pc_o3d).transform(reg_p2l.transformation)
    if np.linalg.norm(corrected_shape.get_center() - masked_pcd.get_center()) > 0.3:
        print("WARNING: icp might not have converged, skipping")
        return PostprObjPred(pred.pc_o3d, pred.grasp_poses)
    corrected_grasps = reg_p2l.transformation @ pred.grasp_poses
    return PostprObjPred(corrected_shape, corrected_grasps)


def postprocess_grasps(
    pred: PostprObjPred,
    full_pcd_fcl: fcl.CollisionObject,
    num_grasps: int,
    joint_state: np.ndarray = None,
    wTbase: sm.SE3 = sm.SE3(),
    wTcam: sm.SE3 = sm.SE3(),
) -> PostprObjPred:
    # Filter grasp not aligned with camera
    pred.grasp_poses = pred.grasp_poses[np.where(pred.grasp_poses[:, 2, 2] > 0)]
    grasp_torque_scores = np.array(
        [torque_score_fun(grasp, pred.pc_o3d.get_center()) for grasp in pred.grasp_poses]
    )
    sorted_indeces = np.argsort(grasp_torque_scores)[::-1]

    best_grasps = []
    # Take the first num_grasps elements not in collision with the scene
    for i in sorted_indeces:
        # If we have enough grasps, stop
        if len(best_grasps) >= num_grasps:
            break
        # If grasp is unreachable, skip it TODO
        # if joint_state is not None and not FRANKA_KIN.goal_reachable(
        #     pred.grasp_poses[i], joint_state, wTbase, wTcam
        # ):
        #     continue
        # If grasp is in collision with the scene, skip it
        if not GRIPPER_FCL.check_valid_grasp(
            sm.SE3(pred.grasp_poses[i], check=False), full_pcd_fcl
        ):
            continue
        # Else, add grasp to the list
        best_grasps.append(pred.grasp_poses[i])
    pred.grasp_poses = np.array(best_grasps)
    return pred


def torque_score_fun(grasp_pose: np.ndarray, center: np.ndarray) -> float:
    """
    grasp_pose: 4x4, in hand_frame
    """
    grasp_normal = grasp_pose[:3, 2]
    grasp_position = grasp_pose[:3, 3]
    grasp_point = grasp_position + 0.1034 * grasp_normal  # Translate to contact point
    grasp_torque = np.cross(center - grasp_point, grasp_normal * 9.81)
    # just consider torque around the y-axis
    score = -np.abs(grasp_torque[1])
    return score
