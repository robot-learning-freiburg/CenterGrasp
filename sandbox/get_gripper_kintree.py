import sapien.core as sapien
from centergrasp.configs import Directories

# Init Sapien
engine = sapien.Engine()
scene = engine.create_scene()
loader = scene.create_urdf_loader()
loader.fix_root_link = True
urdf_path = Directories.FRANKA / "hand.urdf"
gripper = loader.load(str(urdf_path))
gripper.set_qpos([0.4, 0.4])
scene.step()

links = gripper.get_links()
for link in links:
    print(link.name)
    print(link.get_pose().to_transformation_matrix())
