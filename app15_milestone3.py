# CHNAGE DIRECTORY ------               cd /home/arjunmenon/Desktop/ME437/foamcutout/
# ENABLE A PYTHON ENVIRONMENT ------    source foamenv/bin/activate

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import cv2
import plotly.graph_objects as go
from scipy.spatial import Delaunay
from stl import mesh
import io
import tempfile
import matplotlib.pyplot as plt
import ezdxf  # <-- DXF support

st.set_page_config(layout="wide", page_title="Tool Foam Cutout Designer")

UI_MAX_SIZE = 600  # px, for UI/canvas ops

def get_average_rgb(pixels):
    arr = np.array(pixels)
    return tuple(np.mean(arr, axis=0, dtype=int))

def apply_chroma_key(image, avg_rgb, tol):
    arr = np.array(image)
    lower = np.maximum(0, np.array(avg_rgb) - tol)
    upper = np.minimum(255, np.array(avg_rgb) + tol)
    mask = cv2.inRange(arr, lower, upper)
    fg_mask = cv2.bitwise_not(mask)
    return fg_mask

def dilate_mask(mask, thickness):
    k = max(1, int(thickness/2))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1))
    return cv2.dilate(mask, kernel)

def draw_finger_holes(mask, holes):
    mask = mask.copy()
    for x, y, r in holes:
        if r > 0:
            cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
    return mask

def show_mask_overlay_with_holes(image, mask, holes):
    overlay = np.array(image.copy())
    overlay[mask==255] = [255,0,0]
    for x, y, r in holes:
        if r > 0:
            cv2.circle(overlay, (int(x), int(y)), int(r), (255,0,0), -1)
            cv2.circle(overlay, (int(x), int(y)), int(r), (255,255,0), 2)
    return Image.fromarray(overlay)

def find_cavity_contours(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

def triangulate_contour(pts):
    tri = Delaunay(pts)
    vertices = pts
    faces = tri.simplices
    return vertices, faces

@st.cache_data(show_spinner=False)
def plot_3d_foam_matplotlib(base_w, base_h, base_d, cav_d, cav_off, cavity_contours, scale_x, scale_y):
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111, projection='3d')
    verts = np.array([
        [0,0,0],[base_w,0,0],[base_w,base_h,0],[0,base_h,0],
        [0,0,base_d],[base_w,0,base_d],[base_w,base_h,base_d],[0,base_h,base_d]
    ])
    edges = [(0,1),(1,2),(2,3),(3,0),
             (4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    for e in edges:
        ax.plot3D(*zip(*verts[list(e)]), color="gray", linestyle="--")
    start_z = base_d - cav_off
    if start_z > base_d:
        start_z = base_d
    if start_z < 0:
        start_z = 0
    bottom_z = start_z - cav_d
    if bottom_z < 0:
        bottom_z = 0
    if bottom_z > base_d:
        bottom_z = base_d
    for cnt in cavity_contours:
        pts = cnt.reshape(-1,2).astype(float)
        pts[:,0] = pts[:,0] * scale_x
        pts[:,1] = pts[:,1] * scale_y
        xs, ys = pts[:,0], base_h - pts[:,1]
        xs = np.append(xs, xs[0])
        ys = np.append(ys, ys[0])
        ax.plot(xs, ys, start_z, color='red')
        ax.plot(xs, ys, bottom_z, color='red')
        for i in range(0, len(pts), max(1,len(pts)//20)):
            ax.plot([pts[i,0], pts[i,0]],
                    [base_h-pts[i,1], base_h-pts[i,1]], [start_z, bottom_z], color='red', linewidth=0.5)
    ax.set_xlabel(f"X (mm)")
    ax.set_ylabel(f"Y (mm)")
    ax.set_zlabel(f"Z (mm)")
    ax.set_xlim(0, base_w)
    ax.set_ylim(0, base_h)
    ax.set_zlim(0, base_d)
    ax.set_box_aspect([base_w, base_h, base_d])
    st.pyplot(fig)

@st.cache_data(show_spinner=False)
def plot_3d_foam_plotly_mesh3d(base_w, base_h, base_d, cav_d, cav_off, cavity_contours, scale_x, scale_y):
    fig = go.Figure()
    x = [0, base_w, base_w, 0, 0, base_w, base_w, 0]
    y = [0, 0, base_h, base_h, 0, 0, base_h, base_h]
    z = [0, 0, 0, 0, base_d, base_d, base_d, base_d]
    lines = [
        [0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7]
    ]
    for l in lines:
        fig.add_trace(go.Scatter3d(
            x=[x[l[0]], x[l[1]]],
            y=[y[l[0]], y[l[1]]],
            z=[z[l[0]], z[l[1]]],
            mode='lines',
            line=dict(color='gray', width=2),
            showlegend=False
        ))

    start_z = base_d - cav_off
    if start_z > base_d:
        start_z = base_d
    if start_z < 0:
        start_z = 0
    bottom_z = start_z - cav_d
    if bottom_z < 0:
        bottom_z = 0
    if bottom_z > base_d:
        bottom_z = base_d

    for cnt in cavity_contours:
        pts2d = cnt.reshape(-1,2).astype(float)
        pts2d[:,0] = pts2d[:,0] * scale_x
        pts2d[:,1] = pts2d[:,1] * scale_y
        pts2d[:,1] = base_h - pts2d[:,1]
        N = len(pts2d)
        if N >= 3:
            verts2d, faces2d = triangulate_contour(pts2d)
            fig.add_trace(go.Mesh3d(
                x=verts2d[:,0], y=verts2d[:,1], z=np.full(len(verts2d), start_z),
                i=faces2d[:,0], j=faces2d[:,1], k=faces2d[:,2],
                color='red', opacity=0.3, name='top'
            ))
            fig.add_trace(go.Mesh3d(
                x=verts2d[:,0], y=verts2d[:,1], z=np.full(len(verts2d), bottom_z),
                i=faces2d[:,0], j=faces2d[:,1], k=faces2d[:,2],
                color='red', opacity=0.3, name='bottom'
            ))
            for i in range(N):
                j = (i+1)%N
                quad_x = [pts2d[i,0], pts2d[j,0], pts2d[j,0], pts2d[i,0]]
                quad_y = [pts2d[i,1], pts2d[j,1], pts2d[j,1], pts2d[i,1]]
                quad_z = [start_z, start_z, bottom_z, bottom_z]
                fig.add_trace(go.Mesh3d(
                    x=quad_x, y=quad_y, z=quad_z,
                    i=[0], j=[1], k=[2],
                    color='red', opacity=0.3, name='side1'
                ))
                fig.add_trace(go.Mesh3d(
                    x=quad_x, y=quad_y, z=quad_z,
                    i=[0], j=[2], k=[3],
                    color='red', opacity=0.3, name='side2'
                ))
        xs, ys = pts2d[:,0], pts2d[:,1]
        xs = np.append(xs, xs[0])
        ys = np.append(ys, ys[0])
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=[start_z]*len(xs),
            mode='lines', line=dict(color='red', width=4), showlegend=False
        ))
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=[bottom_z]*len(xs),
            mode='lines', line=dict(color='red', width=4), showlegend=False
        ))
        for i in range(len(xs)-1):
            fig.add_trace(go.Scatter3d(
                x=[xs[i], xs[i]], y=[ys[i], ys[i]], z=[start_z, bottom_z],
                mode="lines", line=dict(color='red', width=2), showlegend=False
            ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data',
        ),
        margin=dict(l=0,r=0,b=0,t=0),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data(show_spinner=False)
def export_stl_cached(contours, hierarchy, base_w, base_h, base_d, cav_d, cav_off, scale_x, scale_y):
    verts = [
        [0,0,0],[base_w,0,0],[base_w,base_h,0],[0,base_h,0],
        [0,0,base_d],[base_w,0,base_d],[base_w,base_h,base_d],[0,base_h,base_d]
    ]
    faces = [
        [4,5,6],[4,6,7],[0,3,2],[0,2,1],[3,7,6],[3,6,2],
        [0,1,5],[0,5,4],[1,2,6],[1,6,5],[0,4,7],[0,7,3]
    ]
    offset = len(verts)
    h = hierarchy[0] if hierarchy is not None and hierarchy.ndim==3 else hierarchy

    start_z = base_d - cav_off
    if start_z > base_d:
        start_z = base_d
    if start_z < 0:
        start_z = 0
    bottom_z = start_z - cav_d
    if bottom_z < 0:
        bottom_z = 0
    if bottom_z > base_d:
        bottom_z = base_d

    def add_contour(cnt, z1, z2, is_outer):
        nonlocal verts, faces, offset
        pts = cnt.reshape(-1,2).astype(float)
        pts[:,0] = pts[:,0] * scale_x
        pts[:,1] = pts[:,1] * scale_y
        pts[:,1] = base_h - pts[:,1]
        n = len(pts)
        if n<3: return
        for x,y in pts:
            verts.append([x,y,z1])
            verts.append([x,y,z2])
        for i in range(n):
            j = (i+1)%n
            v0, vb0 = offset+2*i, offset+2*i+1
            v1, vb1 = offset+2*j, offset+2*j+1
            faces.append([v0, vb1, vb0])
            faces.append([v0, v1, vb1])
        for i in range(1,n-1):
            t0, t1, t2 = offset, offset+2*i, offset+2*(i+1)
            b0, b1, b2 = t0+1, t1+1, t2+1
            if is_outer:
                faces.append([t0, t2, t1])
                faces.append([b0, b1, b2])
            else:
                faces.append([t0, t1, t2])
                faces.append([b0, b2, b1])
        offset += 2*n

    for idx,cnt in enumerate(contours):
        if cv2.contourArea(cnt)<1: continue
        outer = (h[idx][3]==-1)
        add_contour(cnt, start_z, bottom_z, outer)
    verts = np.array(verts)
    faces = np.array(faces)
    m = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, fidx in enumerate(faces):
        m.vectors[i] = verts[fidx]
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=True) as tmpfile:
        m.save(tmpfile.name)
        tmpfile.seek(0)
        return tmpfile.read()

def export_dxf_from_contours(contours, scale_x, scale_y, fname="cutout.dxf"):
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    for cnt in contours:
        pts = cnt.reshape(-1, 2).astype(float)
        pts[:,0] = pts[:,0] * scale_x
        pts[:,1] = pts[:,1] * scale_y
        points = [(float(x), float(y)) for x, y in pts]
        if len(points) > 1:
            msp.add_lwpolyline(points, close=True)
    # Write to a StringIO (text) and encode to bytes for download
    import io
    dxf_text = io.StringIO()
    doc.write(dxf_text)
    dxf_bytes = dxf_text.getvalue().encode("utf-8")
    return dxf_bytes

def sync_slider_and_input(label, min_value, max_value, value, key, step=1):
    slider_key = f"{key}_slider"
    input_key = f"{key}_input"
    v = st.session_state.get(input_key, value)
    s1, s2 = st.columns([4,1])
    with s1:
        slider_val = st.slider(label, min_value, max_value, int(v), key=slider_key, step=step, format="%d")
    with s2:
        input_val = st.number_input("", min_value, max_value, int(slider_val), key=input_key, step=step)
    if slider_val != st.session_state[input_key]:
        st.session_state[input_key] = slider_val
        input_val = slider_val
    elif input_val != st.session_state[slider_key]:
        st.session_state[slider_key] = input_val
        slider_val = input_val
    return slider_val

if "last_width" not in st.session_state:
    st.session_state.last_width = 200
if "last_length" not in st.session_state:
    st.session_state.last_length = 200
if "bg_canvas_key" not in st.session_state:
    st.session_state.bg_canvas_key = 0
if "hole_canvas_key" not in st.session_state:
    st.session_state.hole_canvas_key = 0
if "holes" not in st.session_state:
    st.session_state.holes = []
if "finalized_holes" not in st.session_state:
    st.session_state.finalized_holes = []
if "finalized" not in st.session_state:
    st.session_state.finalized = False
if "erase_paths" not in st.session_state:
    st.session_state.erase_paths = []
if "erase_brush_size" not in st.session_state:
    st.session_state.erase_brush_size = 20

st.title("Embry-Riddle Aeronautical University - Mechanical Engineering Department")
st.title("Senior Design : Project Tracker : Automatic Tool Cutout Designer")

col1, col2 = st.columns([2, 1])
with col2:
    width = st.number_input("Desired Width (mm)", min_value=1, max_value=5000, value=200, step=1, key="width_input")
    length = st.number_input("Desired Length (mm)", min_value=1, max_value=5000, value=200, step=1, key="length_input")
    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp", "bmp", "tiff", "gif"]
    )

size_changed = (st.session_state.last_width != width) or (st.session_state.last_length != length)
if size_changed:
    st.session_state.bg_pixels = []
    st.session_state.holes = []
    st.session_state.finalized_holes = []
    st.session_state.finalized = False
    st.session_state.bg_canvas_key += 1
    st.session_state.hole_canvas_key += 1
    st.session_state.erase_paths = []
st.session_state.last_width = width
st.session_state.last_length = length

if uploaded:
    orig_image = Image.open(uploaded)
    if orig_image.mode in ("RGBA", "LA") or (orig_image.mode == "P" and "transparency" in orig_image.info):
        background = Image.new("RGBA", orig_image.size, (255,255,255,255))
        orig_image = Image.alpha_composite(background, orig_image.convert("RGBA")).convert("RGB")
    else:
        orig_image = orig_image.convert("RGB")
    orig_w, orig_h = orig_image.width, orig_image.height
    scale = min(UI_MAX_SIZE / max(orig_w, orig_h), 1.0)
    ui_w, ui_h = int(orig_w * scale), int(orig_h * scale)
    ui_image = orig_image.resize((ui_w, ui_h), Image.LANCZOS)
else:
    st.info("Upload an image to get started.")
    st.stop()

col1, col2 = st.columns([2,1])
with col1:
    st.image(ui_image, caption=f"UI Image: {ui_w} x {ui_h} px", use_column_width=True)

st.markdown("### 2. Select Background Pixels (Draw Points on Image)")
canvas_col, slider_col = st.columns([2,1])
with canvas_col:
    if "bg_pixels" not in st.session_state:
        st.session_state.bg_pixels = []
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=5,
        background_image=ui_image,
        update_streamlit=True,
        height=ui_h,
        width=ui_w,
        drawing_mode="point",
        point_display_radius=6,
        key=f"canvas_bg_{ui_w}_{ui_h}_{st.session_state.bg_canvas_key}"
    )
    bg_points = []
    if canvas_result.json_data and "objects" in canvas_result.json_data:
        bg_points = [
            (int(obj["left"]), int(obj["top"]))
            for obj in canvas_result.json_data["objects"] if obj["type"] == "circle"
        ]
        st.session_state.bg_pixels = [ui_image.getpixel((x, y)) for x, y in bg_points if 0 <= x < ui_w and 0 <= y < ui_h]

with slider_col:
    if st.button("Clear Selected Background Pixels"):
        st.session_state.bg_pixels = []
        st.session_state.bg_canvas_key += 1
        st.experimental_rerun()

    if st.session_state.bg_pixels:
        st.write(f"Selected: {len(st.session_state.bg_pixels)} pixels")
        avg_rgb = get_average_rgb(st.session_state.bg_pixels)
        st.write(f"Average RGB: {avg_rgb}")
    else:
        st.warning("No background pixels selected yet.")
        st.stop()

st.markdown("### 3. Chroma Key and Border/Finger Holes")
canvas_col, slider_col = st.columns([2,1])
with slider_col:
    tol = sync_slider_and_input("Chroma Key Tolerance", 1, 80, 25, "tol")
    init_thickness = sync_slider_and_input("Border Thickness (in mm)", 0, 100, 6, "init_thickness")
with canvas_col:
    mask = apply_chroma_key(ui_image, avg_rgb, tol)
    border_mask = dilate_mask(mask, init_thickness)

st.markdown("#### Add Finger Holes or Brush Erase (Draw Circles or Freehand Erase)")
canvas_col, slider_col = st.columns([2,1])

with slider_col:
    tool_mode = st.radio("Drawing tool", ["Circle (add hole)", "Brush (erase blemish)"], horizontal=True)
    if tool_mode == "Brush (erase blemish)":
        st.session_state.erase_brush_size = st.slider("Brush size (px)", 5, 50, st.session_state.erase_brush_size, key="brush_size")
    if st.button("Clear All Holes & Brush Erase"):
        st.session_state.holes = []
        st.session_state.finalized_holes = []
        st.session_state.finalized = False
        st.session_state.erase_paths = []
        st.session_state.hole_canvas_key += 1
        st.experimental_rerun()
    if st.session_state.holes:
        st.write("Adjust radii for finger holes (set to 0 to delete):")
        for i, (x, y, r) in enumerate(st.session_state.holes):
            st.session_state.holes[i] = (
                x, y,
                sync_slider_and_input(f"Hole {i+1} radius", 0, max(ui_w, ui_h)//2, r, f"hole_radius_{i}")
            )
    if st.button("Finalize"):
        st.session_state.finalized_holes = list(st.session_state.holes)
        st.session_state.finalized = True
        st.success("Finalized! See updated 3D preview below.")

with canvas_col:
    holes_for_mask = [h for h in st.session_state.holes if h[2] > 0]
    combined_mask = cv2.bitwise_or(
        border_mask,
        draw_finger_holes(np.zeros_like(border_mask), holes_for_mask)
    )
    hole_canvas_background = show_mask_overlay_with_holes(ui_image, border_mask, holes_for_mask)

    if tool_mode == "Circle (add hole)":
        canvas_result_holes = st_canvas(
            fill_color="rgba(255,255,0,0.4)",
            stroke_width=5,
            background_image=hole_canvas_background,
            update_streamlit=True,
            height=ui_h,
            width=ui_w,
            drawing_mode="circle",
            key=f"canvas_holes_{ui_w}_{ui_h}_{st.session_state.hole_canvas_key}"
        )
        new_holes = []
        if canvas_result_holes.json_data and "objects" in canvas_result_holes.json_data:
            for obj in canvas_result_holes.json_data["objects"]:
                if obj["type"] == "circle":
                    x = int(obj["left"])
                    y = int(obj["top"])
                    existing_idx = next((i for i, h in enumerate(st.session_state.holes)
                                         if abs(h[0]-x) < 4 and abs(h[1]-y) < 4), None)
                    if existing_idx is not None:
                        r = st.session_state.holes[existing_idx][2]
                    else:
                        r = int(obj["radius"])
                    new_holes.append((x, y, r))
        st.session_state.holes = new_holes
    else:
        canvas_result_brush = st_canvas(
            fill_color="rgba(0,0,0,0.0)",
            stroke_width=st.session_state.erase_brush_size,
            background_image=hole_canvas_background,
            update_streamlit=True,
            height=ui_h,
            width=ui_w,
            drawing_mode="freedraw",
            key=f"canvas_erase_{ui_w}_{ui_h}_{st.session_state.hole_canvas_key}"
        )
        if canvas_result_brush.json_data and "objects" in canvas_result_brush.json_data:
            erase_paths = []
            for obj in canvas_result_brush.json_data["objects"]:
                if obj["type"] == "path":
                    path_points = obj.get("path", [])
                    for point in path_points:
                        if len(point) >= 3:
                            x, y = int(point[1]), int(point[2])
                            erase_paths.append((x, y, st.session_state.erase_brush_size))
            st.session_state.erase_paths += erase_paths

    erase_mask = np.zeros_like(combined_mask)
    for (x, y, brush_size) in st.session_state.erase_paths:
        cv2.circle(erase_mask, (x, y), brush_size//2, 255, -1)
    combined_mask[erase_mask==255] = 0

if st.session_state.finalized:
    st.markdown("### 4. 3D Preview & STL/DXF Export")
    col3d1, col3d2, col3dctrl = st.columns([1.5,1.5,1])
    final_mask = cv2.bitwise_or(
        border_mask,
        draw_finger_holes(np.zeros_like(border_mask), [h for h in st.session_state.finalized_holes if h[2] > 0])
    )
    erase_mask = np.zeros_like(final_mask)
    for (x, y, brush_size) in st.session_state.erase_paths:
        cv2.circle(erase_mask, (x, y), brush_size//2, 255, -1)
    final_mask[erase_mask==255] = 0

    export_mask = cv2.resize(final_mask, (ui_w, ui_h), interpolation=cv2.INTER_NEAREST)
    export_h, export_w = export_mask.shape
    scale_x = width / export_w
    scale_y = length / export_h
    cavity_contours, hierarchy = find_cavity_contours(export_mask)

    with col3dctrl:
        base_depth = sync_slider_and_input("Foam Base Depth (mm)", 1, 5000, 30, "base_depth")
        cavity_depth = sync_slider_and_input("Cavity Depth (mm)", 1, 5000, 15, "cavity_depth")
        cavity_offset = sync_slider_and_input("Cavity Offset (mm)", 0, 5000, 0, "cavity_offset")

        stl_file_name = st.text_input(
            "STL File Name - change by the textbox below - ", value="tool_foam_cutout.stl", key="stl_file_name"
        )
        if not stl_file_name.lower().endswith(".stl"):
            stl_file_name += ".stl"

    with col3d1:
        st.markdown("**Matplotlib 3D View**")
        if cavity_contours is not None and len(cavity_contours) > 0:
            plot_3d_foam_matplotlib(width, length, base_depth, cavity_depth, cavity_offset, cavity_contours, scale_x, scale_y)
        else:
            st.warning("No cavity contours found for 3D preview.")

    with col3d2:
        st.markdown("**Plotly 3D View (Mesh3d, Interactive)**")
        if cavity_contours is not None and len(cavity_contours) > 0:
            plot_3d_foam_plotly_mesh3d(width, length, base_depth, cavity_depth, cavity_offset, cavity_contours, scale_x, scale_y)
        else:
            st.warning("No cavity contours found for 3D preview.")

    with col3dctrl:
        if st.button("Export STL - for 3d-Printing and CAD applications"):
            if not cavity_contours or len(cavity_contours)==0:
                st.error("No valid cavity for export.")
            else:
                st.info("Exporting, please wait...")
                st.download_button(
                    label="Download STL",
                    data=export_stl_cached(cavity_contours, hierarchy, width, length, base_depth, cavity_depth, cavity_offset, scale_x, scale_y),
                    file_name=stl_file_name
                )

        if st.button("Export DXF - for laser Cutting "):
            if not cavity_contours or len(cavity_contours)==0:
                st.error("No valid cavity for export.")
            else:
                st.info("Exporting DXF, please wait...")
                dxf_file_name = stl_file_name.rsplit('.', 1)[0] + ".dxf"
                st.download_button(
                    label="Download DXF",
                    data=export_dxf_from_contours(cavity_contours, scale_x, scale_y, dxf_file_name),
                    file_name=dxf_file_name
                )
