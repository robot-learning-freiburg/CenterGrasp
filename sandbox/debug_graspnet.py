import open3d as o3d
from graspnetAPI.utils.utils import create_mesh_box

box_o3d = create_mesh_box(0.1, 0.2, 0.3)
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

# Visualize
o3d.visualization.draw_geometries([box_o3d, frame])

# width = grasp width
# depth + 0.02 = length of the fingers
# height = thickness of the fingers
