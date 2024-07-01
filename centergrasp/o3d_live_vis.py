import open3d as o3d

try:
    from pynput import keyboard
except ImportError:
    print("pynput not installed, cannot listen to esc key")


# This class is in a separate file to avoid circular imports
class O3dLiveVisualizer:
    def __init__(self, vis_flag) -> None:
        # Initialize
        self.vis_flag = vis_flag
        if not vis_flag:
            return
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        self.vis = vis

        # Main frame
        frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.add_geometries(frame_mesh)

        # Listen to esc key
        self.waiting_esc = True
        key_listener = keyboard.Listener(on_press=self.on_press)
        key_listener.start()
        return

    def add_and_render(self, geoms):
        if not self.vis_flag:
            return
        self.add_geometries(geoms)
        self.update_renderer()
        return

    def update_and_render(self, geoms):
        if not self.vis_flag:
            return
        self.update_geometries(geoms)
        self.update_renderer()
        return

    def remove_and_render(self, geoms):
        if not self.vis_flag:
            return
        self.remove_geometries(geoms)
        self.update_renderer()
        return

    def add_geometries(self, geoms):
        if isinstance(geoms, list):
            for geom in geoms:
                self.vis.add_geometry(geom)
        elif isinstance(geoms, dict):
            for geom in geoms.values():
                self.vis.add_geometry(geom)
        else:
            self.vis.add_geometry(geoms)
        return

    def update_geometries(self, geoms):
        if isinstance(geoms, list):
            for geom in geoms:
                self.vis.update_geometry(geom)
        elif isinstance(geoms, dict):
            for geom in geoms.values():
                self.vis.update_geometry(geom)
        else:
            self.vis.update_geometry(geoms)
        return

    def remove_geometries(self, geoms):
        if isinstance(geoms, list):
            for geom in geoms:
                self.vis.remove_geometry(geom)
        elif isinstance(geoms, dict):
            for geom in geoms.values():
                self.vis.remove_geometry(geom)
        else:
            self.vis.remove_geometry(geoms)
        return

    def update_renderer(self):
        while self.waiting_esc:
            self.vis.poll_events()
            self.vis.update_renderer()
        self.waiting_esc = True
        return

    def on_press(self, key):
        try:
            if key == keyboard.Key.esc:
                self.waiting_esc = False
        except (KeyError, AttributeError):
            pass
        return
