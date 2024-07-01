import numpy as np
from PIL import Image
from centergrasp.configs import Directories
import sapien.core as sapien
from sapien.utils.viewer import Viewer
import open3d as o3d

raytracing = False
texture_name = "Bricks082A"
texture_path = Directories.TEXTURES / texture_name
mesh_path = Directories.GIGA_REPO / "data/urdfs/packed/test/CoffeeCookies_800_tex_visual.obj"

# Raytracing
if raytracing:
    sapien.render_config.camera_shader_dir = "rt"
    sapien.render_config.viewer_shader_dir = "rt"
    sapien.render_config.rt_samples_per_pixel = 32
    sapien.render_config.rt_use_denoiser = True

# Scene
engine = sapien.Engine()
renderer = sapien.SapienRenderer()
engine.set_renderer(renderer)
scene = engine.create_scene()
scene.set_timestep(1 / 100.0)
viewer = Viewer(renderer)
viewer.set_scene(scene)
viewer.set_camera_xyz(x=-0.2, y=0.0, z=0.0)
viewer.toggle_axes(False)

# Light
scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)

# Material
material = renderer.create_material()
material.set_diffuse_texture_from_file(str(texture_path / f"{texture_name}_2K_Color.jpg"))
material.set_roughness_texture_from_file(str(texture_path / f"{texture_name}_2K_Roughness.jpg"))
material.set_normal_texture_from_file(str(texture_path / f"{texture_name}_2K_NormalGL.jpg"))

# Box
actor_builder = scene.create_actor_builder()
actor_builder.add_box_collision(half_size=[0.05, 0.05, 0.05])
actor_builder.add_box_visual(half_size=[0.05, 0.05, 0.05], material=material)
box = actor_builder.build_kinematic(name="box")
box.set_pose(sapien.Pose(p=[0.5, -0.1, 0]))

# Get UV Atlas (only available for tensor mesh)
o3d_mesh = o3d.t.io.read_triangle_mesh(str(mesh_path))
o3d_mesh.compute_uvatlas()
vertices = o3d_mesh.vertex.positions.numpy()  # shape: (200, 3)
triangles = o3d_mesh.triangle.indices.numpy()  # shape: (396, 3)
triangle_uvs = o3d_mesh.triangle.texture_uvs.numpy()  # shape: (396, 3, 2)
_, ids = np.unique(triangles, return_index=True)
vertex_uvs = triangle_uvs.reshape(-1, 2)[ids]  # shape: (200, 2)

# Render Mesh
render_mesh = renderer.create_mesh(vertices, triangles)
render_mesh.set_uvs(vertex_uvs)

# Mesh
actor_builder = scene.create_actor_builder()
actor_builder.add_collision_from_file(str(mesh_path))
# actor_builder.add_visual_from_file(str(mesh_path), material=material)
actor_builder.add_visual_from_mesh(render_mesh, material=material)
mesh = actor_builder.build_kinematic(name="mesh")
mesh.set_pose(sapien.Pose(p=[0.5, 0.1, 0]))

# Camera
near, far = 0.1, 100
width, height = 640, 480
camera = scene.add_camera(
    name="camera",
    width=width,
    height=height,
    fovy=np.deg2rad(35),
    near=near,
    far=far,
)
camera.set_pose(sapien.Pose(p=[0, 0, 0.25], q=[0.9659258, 0, 0.258819, 0]))

# Render
scene.step()  # make everything set
scene.update_render()
camera.take_picture()
rgba = camera.get_color_rgba()  # [H, W, 4]
rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
rgba_pil = Image.fromarray(rgba_img)
rgba_pil.save("color.png")

while not viewer.closed:  # Press key q to quit
    scene.step()
    scene.update_render()
    viewer.render()
