import os
os.environ["XDG_SESSION_TYPE"] = "x11"
os.environ.pop("WAYLAND_DISPLAY", None)

import json
import datetime
import numpy as np
import pandas as pd
import open3d as o3d

# --- Load data ---
df = pd.read_csv(
    "map_mgo12/tracking.csv",
    comment="#",
    header=None,
    names=["ts", "tx", "ty", "tz", "qw", "qx", "qy", "qz"],
)

with open("map_mgo12/opt_map_filtered.json") as f:
    map_data = json.load(f)

with open("map_mgo12/covisibility_graph.json") as f:
    cov_graph = json.load(f)

landmarks = np.array([lm["p_w"] for lm in map_data["landmarks"]])
traj = df[["tx", "ty", "tz"]].to_numpy()

# Build timestamp -> trajectory index lookup
ts_to_idx = {ts: i for i, ts in enumerate(df["ts"])}

def edges_to_lineset(edges, color):
    pts = []
    lines = []
    for ts0, ts1 in edges:
        i0 = ts_to_idx.get(ts0)
        i1 = ts_to_idx.get(ts1)
        if i0 is None or i1 is None:
            continue
        idx = len(pts)
        pts.append(traj[i0])
        pts.append(traj[i1])
        lines.append([idx, idx + 1])
    if not pts:
        return None
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(pts))
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.paint_uniform_color(color)
    return ls

# --- Point cloud ---
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(landmarks)
pcd.paint_uniform_color([0.1, 0.2, 0.8])

# --- Trajectory (dim grey) ---
n = len(traj)
indices = [[i, i + 1] for i in range(n - 1)]
lineset = o3d.geometry.LineSet()
lineset.points = o3d.utility.Vector3dVector(traj)
lineset.lines = o3d.utility.Vector2iVector(indices)
lineset.paint_uniform_color([0.7, 0.7, 0.78])  # lightened to simulate ~50% transparency on white bg

# --- Covisibility graph edges (muted steel blue) ---
covis_lineset = edges_to_lineset(
    cov_graph["high_covisibility_edges"], [0.3, 0.55, 0.75]
)

# --- Loop closure edges (bright yellow-orange) ---
loop_lineset = edges_to_lineset(
    cov_graph["loop_closure_edges"], [0.4, 0.8, 0.4]
)

# --- Start / end markers as small spheres ---
def sphere_at(center, radius=0.05, color=(0, 1, 0)):
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    s.translate(center)
    s.paint_uniform_color(color)
    s.compute_vertex_normals()
    return s

#start_sphere = sphere_at(traj[0], color=(0.2, 0.9, 0.2))
#end_sphere = sphere_at(traj[-1], color=(0.9, 0.2, 0.2))

# --- Visualize ---
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(window_name="Trajectory + Map", width=3840, height=2160)

vis.add_geometry(pcd)
vis.add_geometry(lineset)
#if covis_lineset is not None:
#    vis.add_geometry(covis_lineset)
#if loop_lineset is not None:
#    vis.add_geometry(loop_lineset)
#vis.add_geometry(start_sphere)
#vis.add_geometry(end_sphere)

opt = vis.get_render_option()
#opt.background_color = np.array([0.05, 0.05, 0.08])
opt.point_size = 1.7

def _save_screenshot(vis):
    path = f"screenshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    vis.capture_screen_image(path, do_render=True)
    print(f"Saved: {path}")
    return False

vis.register_key_callback(ord("S"), _save_screenshot)

vis.run()

# Auto-save on close
_save_screenshot(vis)

vis.destroy_window()
