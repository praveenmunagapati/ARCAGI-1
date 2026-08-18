"""
MATHX ARC-AGI-1 GPU-ACCELERATED REASONING & BENCHMARK ENGINE
Native execution on NVIDIA GeForce MX330 GPU via Vulkan / WGPU Compute Shaders.
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Optional, Any
import numpy as np
import wgpu

Grid = np.ndarray

def G(x) -> Grid:
    return np.asarray(x, dtype=np.int32)

def exact(a: Optional[Grid], b: Optional[Grid]) -> bool:
    if a is None or b is None:
        return False
    return a.shape == b.shape and np.array_equal(a, b)


# ============================================================
# GPU COMPUTE CONTEXT & WGSL SHADER PIPELINE (NVIDIA MX330)
# ============================================================

class GPUComputeEngine:
    _instance = None

    @classmethod
    def get(cls) -> GPUComputeEngine:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        try:
            self.adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
            self.device = self.adapter.request_device_sync()
            self.device_name = self.adapter.summary
            self.available = True
            self._compile_kernels()
            print(f"[GPU] NVIDIA GPU Compute Engine initialized: {self.device_name}")
        except Exception as e:
            self.available = False
            self.device_name = f"CPU Vectorized Fallback ({e})"
            print(f"[GPU] Warning: GPU initialization fallback: {e}")

    def _compile_kernels(self):
        """Compile unified WGSL compute shader for ARC grid operations on NVIDIA MX330."""
        self.shader_code = """
        struct KernelParams {
            op_type: u32,
            height: u32,
            width: u32,
            param0: u32,
        };

        @group(0) @binding(0) var<uniform> params: KernelParams;
        @group(0) @binding(1) var<storage, read> input_grid: array<i32>;
        @group(0) @binding(2) var<storage, read_write> output_grid: array<i32>;
        @group(0) @binding(3) var<storage, read> palette_lut: array<i32>;
        @group(0) @binding(4) var<storage, read_write> match_result: array<i32>;

        @compute @workgroup_size(16, 16)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let r = gid.y;
            let c = gid.x;
            let h = params.height;
            let w = params.width;

            if (r >= h || c >= w) {
                return;
            }

            let in_idx = r * w + c;
            let val = input_grid[in_idx];

            if (params.op_type == 0u) {
                // 0: Identity
                output_grid[in_idx] = val;
            } else if (params.op_type == 1u) {
                // 1: Rot 90 CW (w x h output)
                let out_idx = c * h + (h - 1u - r);
                output_grid[out_idx] = val;
            } else if (params.op_type == 2u) {
                // 2: Rot 180 (h x w output)
                let out_idx = (h - 1u - r) * w + (w - 1u - c);
                output_grid[out_idx] = val;
            } else if (params.op_type == 3u) {
                // 3: Rot 270 CW (w x h output)
                let out_idx = (w - 1u - c) * h + r;
                output_grid[out_idx] = val;
            } else if (params.op_type == 4u) {
                // 4: Flip H
                let out_idx = r * w + (w - 1u - c);
                output_grid[out_idx] = val;
            } else if (params.op_type == 5u) {
                // 5: Flip V
                let out_idx = (h - 1u - r) * w + c;
                output_grid[out_idx] = val;
            } else if (params.op_type == 6u) {
                // 6: Transpose
                let out_idx = c * h + r;
                output_grid[out_idx] = val;
            } else if (params.op_type == 7u) {
                // 7: Palette Lookup Map
                if (val >= 0 && val < 10) {
                    output_grid[in_idx] = palette_lut[val];
                } else {
                    output_grid[in_idx] = val;
                }
            } else if (params.op_type == 8u) {
                // 8: Compare Exact Match
                if (input_grid[in_idx] != output_grid[in_idx]) {
                    match_result[0] = 0;
                }
            }
        }
        """
        self.module = self.device.create_shader_module(code=self.shader_code)
        self.bind_group_layout = self.device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
            {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
        ])
        self.pipeline = self.device.create_compute_pipeline(
            layout=self.device.create_pipeline_layout(bind_group_layouts=[self.bind_group_layout]),
            compute={"module": self.module, "entry_point": "main"},
        )
        self.dispatches = 0

    def gpu_transform(self, g: Grid, op_type: int, lut: Optional[np.ndarray] = None) -> Grid:
        """Execute a parallel transformation kernel on the MX330 GPU."""
        if not self.available:
            return self._cpu_fallback_transform(g, op_type, lut)

        h, w = g.shape
        out_h, out_w = (w, h) if op_type in (1, 3, 6) else (h, w)
        out_size = out_h * out_w

        # Prepare uniform params
        params_data = np.array([op_type, h, w, 0], dtype=np.uint32)
        lut_data = np.arange(10, dtype=np.int32) if lut is None else np.asarray(lut, dtype=np.int32)
        match_init = np.array([1], dtype=np.int32)

        in_buf = self.device.create_buffer_with_data(data=g.ravel(), usage=wgpu.BufferUsage.STORAGE)
        out_buf = self.device.create_buffer(size=out_size * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        param_buf = self.device.create_buffer_with_data(data=params_data, usage=wgpu.BufferUsage.UNIFORM)
        lut_buf = self.device.create_buffer_with_data(data=lut_data, usage=wgpu.BufferUsage.STORAGE)
        match_buf = self.device.create_buffer_with_data(data=match_init, usage=wgpu.BufferUsage.STORAGE)

        bind_group = self.device.create_bind_group(
            layout=self.bind_group_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": param_buf}},
                {"binding": 1, "resource": {"buffer": in_buf}},
                {"binding": 2, "resource": {"buffer": out_buf}},
                {"binding": 3, "resource": {"buffer": lut_buf}},
                {"binding": 4, "resource": {"buffer": match_buf}},
            ]
        )

        encoder = self.device.create_command_encoder()
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self.pipeline)
        cpass.set_bind_group(0, bind_group)
        cpass.dispatch_workgroups((w + 15) // 16, (h + 15) // 16)
        cpass.end()

        self.device.queue.submit([encoder.finish()])
        self.dispatches += 1

        res_bytes = self.device.queue.read_buffer(out_buf)
        res_arr = np.frombuffer(res_bytes, dtype=np.int32).reshape((out_h, out_w))
        return res_arr

    def _cpu_fallback_transform(self, g: Grid, op_type: int, lut: Optional[np.ndarray] = None) -> Grid:
        if op_type == 0: return g.copy()
        if op_type == 1: return np.rot90(g, -1)
        if op_type == 2: return np.rot90(g, 2)
        if op_type == 3: return np.rot90(g, 1)
        if op_type == 4: return np.fliplr(g)
        if op_type == 5: return np.flipud(g)
        if op_type == 6: return g.T
        if op_type == 7 and lut is not None:
            out = g.copy()
            for k in range(10):
                out[g == k] = lut[k]
            return out
        return g


# ============================================================
# OBJECT UTILITIES
# ============================================================

@dataclass
class Obj:
    color: int
    cells: list[tuple[int, int]]
    area: int
    bbox: tuple[int, int, int, int]
    h: int
    w: int
    subgrid: Grid

def extract_objects(g: Grid, connectivity: int = 4, bg: int = 0, mono: bool = True) -> list[Obj]:
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    objs = []
    dirs = [(-1,0),(1,0),(0,-1),(0,1)] if connectivity == 4 else [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    for r in range(h):
        for c in range(w):
            if visited[r, c] or g[r, c] == bg:
                continue
            color = int(g[r, c])
            cells = []
            stack = [(r, c)]
            visited[r, c] = True
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                        if mono:
                            if g[nr, nc] == color:
                                visited[nr, nc] = True
                                stack.append((nr, nc))
                        else:
                            if g[nr, nc] != bg:
                                visited[nr, nc] = True
                                stack.append((nr, nc))
            rs = [x[0] for x in cells]
            cs = [x[1] for x in cells]
            min_r, max_r = min(rs), max(rs)
            min_c, max_c = min(cs), max(cs)
            sub = g[min_r:max_r+1, min_c:max_c+1].copy()
            objs.append(Obj(
                color=color,
                cells=cells,
                area=len(cells),
                bbox=(min_r, min_c, max_r, max_c),
                h=max_r - min_r + 1,
                w=max_c - min_c + 1,
                subgrid=sub
            ))
    return objs


# ============================================================
# GPU-ACCELERATED REASONING & SYNTHESIS ENGINE
# ============================================================

class GPUSolverEngine:
    def __init__(self):
        self.gpu = GPUComputeEngine.get()

    def synthesize(self, task: dict) -> Optional[Callable[[Grid], Grid]]:
        train = [(G(ex["input"]), G(ex["output"])) for ex in task["train"]]
        
        # 1. GPU Rigid & Affine Transformations
        sol = self._try_rigid_gpu(train)
        if sol: return sol

        # 2. GPU Palette & Color Remapping
        sol = self._try_palette_gpu(train)
        if sol: return sol

        # 3. Kronecker & Fractal Self-Tiling
        sol = self._try_kronecker(train)
        if sol: return sol

        # 4. Scaling / Zoom / Pixel Repeat
        sol = self._try_scaling(train)
        if sol: return sol

        # 5. Cropping & Bounding Box Extraction
        sol = self._try_crop(train)
        if sol: return sol

        # 6. Multi-Panel & Grid Divider Logic
        sol = self._try_divider_panels(train)
        if sol: return sol

        # 7. Symmetry & Pattern Completion
        sol = self._try_symmetry(train)
        if sol: return sol

        # 8. Enclosed Hole Filling & Morphology
        sol = self._try_hole_filling(train)
        if sol: return sol

        # 9. Gravity & Physics (free fall & obstacle collision)
        sol = self._try_gravity(train)
        if sol: return sol

        # 10. Lines, Raycasting & Orthogonal Crosses
        sol = self._try_lines_and_rays(train)
        if sol: return sol

        # 11. Object Extraction & Filtering
        sol = self._try_object_filtering(train)
        if sol: return sol

        # 12. Object Recoloring by Area Rank / Sequence
        sol = self._try_object_recolor(train)
        if sol: return sol

        # 13. Fill Bounding Boxes of Connected Components
        sol = self._try_bbox_fill(train)
        if sol: return sol

        # 14. Periodic Pattern Extension / Extrapolation
        sol = self._try_periodic_extension(train)
        if sol: return sol

        # 15. Tiling / Pattern Repetition
        sol = self._try_tiling(train)
        if sol: return sol

        # 16. Multi-Step Compositions
        sol = self._try_composition(train)
        if sol: return sol

        return None

    # --- 1. GPU Rigid Transformations ---
    def _try_rigid_gpu(self, train):
        for op_type in (0, 1, 2, 3, 4, 5, 6):
            def make_fn(ot=op_type):
                return lambda g: self.gpu.gpu_transform(g, ot)
            fn = make_fn()
            try:
                if all(exact(fn(inp), out) for inp, out in train):
                    return fn
            except Exception:
                pass
        return None

    # --- 2. GPU Palette Mapping ---
    def _try_palette_gpu(self, train):
        mapping = {}
        consistent = True
        for inp, out in train:
            if inp.shape != out.shape:
                consistent = False; break
            for u in np.unique(inp):
                out_colors = out[inp == u]
                if len(np.unique(out_colors)) != 1:
                    consistent = False; break
                c_target = int(out_colors[0])
                if u in mapping and mapping[u] != c_target:
                    consistent = False; break
                mapping[u] = c_target
            if not consistent:
                break
        if consistent and mapping:
            lut = np.arange(10, dtype=np.int32)
            for k, v in mapping.items():
                if 0 <= k < 10:
                    lut[k] = v
            def fn(g, l=lut):
                return self.gpu.gpu_transform(g, 7, l)
            if all(exact(fn(inp), out) for inp, out in train):
                return fn
        return None

    # --- 3. Kronecker & Fractal ---
    def _try_kronecker(self, train):
        def fn1(g):
            mask = (g > 0).astype(np.int32)
            return np.kron(mask, g)
        if all(exact(fn1(inp), out) for inp, out in train):
            return fn1

        def fn2(g):
            mask = (g > 0).astype(np.int32)
            return np.kron(g, mask)
        if all(exact(fn2(inp), out) for inp, out in train):
            return fn2

        def fn3(g):
            h, w = g.shape
            out = np.zeros((h*h, w*w), dtype=np.int32)
            for r in range(h):
                for c in range(w):
                    if g[r, c] != 0:
                        out[r*h:(r+1)*h, c*w:(c+1)*w] = g
            return out
        try:
            if all(exact(fn3(inp), out) for inp, out in train):
                return fn3
        except Exception:
            pass
        return None

    # --- 4. Scaling / Zoom ---
    def _try_scaling(self, train):
        for sy in (2, 3, 4, 5):
            for sx in (2, 3, 4, 5):
                def make_scale(y_fac=sy, x_fac=sx):
                    return lambda g: np.repeat(np.repeat(g, y_fac, axis=0), x_fac, axis=1)
                fn = make_scale()
                try:
                    if all(exact(fn(inp), out) for inp, out in train):
                        return fn
                except Exception:
                    pass
        return None

    # --- 5. Cropping & Bounding Box ---
    def _try_crop(self, train):
        def crop_non_zero(g):
            rows, cols = np.where(g != 0)
            if len(rows) == 0: return g
            return g[rows.min():rows.max()+1, cols.min():cols.max()+1]
        try:
            if all(exact(crop_non_zero(inp), out) for inp, out in train):
                return crop_non_zero
        except Exception:
            pass

        def crop_most_freq(g):
            counts = [(c, np.count_nonzero(g == c)) for c in np.unique(g) if c != 0]
            if not counts: return g
            counts.sort(key=lambda x: x[1], reverse=True)
            c_target = counts[0][0]
            rows, cols = np.where(g == c_target)
            return g[rows.min():rows.max()+1, cols.min():cols.max()+1]
        try:
            if all(exact(crop_most_freq(inp), out) for inp, out in train):
                return crop_most_freq
        except Exception:
            pass

        for frame_c in range(10):
            def make_frame_crop(fc=frame_c):
                def fn(g):
                    rows, cols = np.where(g == fc)
                    if len(rows) == 0: return g
                    min_r, max_r = rows.min(), rows.max()
                    min_c, max_c = cols.min(), cols.max()
                    if max_r - min_r > 1 and max_c - min_c > 1:
                        return g[min_r+1:max_r, min_c+1:max_c]
                    return g
                return fn
            fn = make_frame_crop()
            try:
                if all(exact(fn(inp), out) for inp, out in train):
                    return fn
            except Exception:
                pass
        return None

    # --- 6. Divider Panels & Multi-panel Logic ---
    def _try_divider_panels(self, train):
        for div_c in range(10):
            def split_panels(g, dc=div_c):
                h, w = g.shape
                div_rows = [r for r in range(h) if np.all(g[r, :] == dc)]
                div_cols = [c for c in range(w) if np.all(g[:, c] == dc)]
                
                row_splits = [-1] + div_rows + [h]
                col_splits = [-1] + div_cols + [w]
                
                panels = []
                for i in range(len(row_splits) - 1):
                    r1, r2 = row_splits[i] + 1, row_splits[i+1]
                    for j in range(len(col_splits) - 1):
                        c1, c2 = col_splits[j] + 1, col_splits[j+1]
                        if r2 > r1 and c2 > c1:
                            panels.append(g[r1:r2, c1:c2])
                return panels

            for op in ("xor", "or", "and", "diff"):
                for recolor in range(10):
                    def make_overlay(dc=div_c, operation=op, rc=recolor):
                        def fn(g):
                            panels = split_panels(g, dc)
                            if len(panels) != 2 or panels[0].shape != panels[1].shape:
                                return None
                            p1, p2 = panels[0], panels[1]
                            res = np.zeros_like(p1)
                            if operation == "xor":
                                mask = (p1 != 0) ^ (p2 != 0)
                            elif operation == "or":
                                mask = (p1 != 0) | (p2 != 0)
                            elif operation == "and":
                                mask = (p1 != 0) & (p2 != 0)
                            elif operation == "diff":
                                mask = (p1 != 0) & (p2 == 0)
                            res[mask] = rc if rc != 0 else (p1[mask] if np.any(p1[mask]) else p2[mask])
                            return res
                        return fn
                    fn = make_overlay()
                    try:
                        if all(exact(fn(inp), out) for inp, out in train):
                            return fn
                    except Exception:
                        pass

            for idx_p in (0, 1, 2, 3, 4, -1):
                def make_panel_idx(dc=div_c, ip=idx_p):
                    def fn(g):
                        panels = split_panels(g, dc)
                        if not panels or abs(ip) >= len(panels): return None
                        return panels[ip]
                    return fn
                fn = make_panel_idx()
                try:
                    if all(exact(fn(inp), out) for inp, out in train):
                        return fn
                except Exception:
                    pass

            for sel in ("max_nonzero", "min_nonzero", "unique"):
                def make_sel(dc=div_c, s=sel):
                    def fn(g):
                        panels = split_panels(g, dc)
                        if not panels: return None
                        if s == "max_nonzero":
                            return max(panels, key=lambda p: np.count_nonzero(p))
                        elif s == "min_nonzero":
                            return min(panels, key=lambda p: np.count_nonzero(p))
                        return panels[0]
                    return fn
                fn = make_sel()
                try:
                    if all(exact(fn(inp), out) for inp, out in train):
                        return fn
                except Exception:
                    pass
        return None

    # --- 7. Symmetry & Pattern Completion ---
    def _try_symmetry(self, train):
        def sym_h_left(g):
            h, w = g.shape
            mid = w // 2
            out = g.copy()
            left = g[:, :mid]
            out[:, w - mid:] = np.fliplr(left)
            return out
        def sym_h_right(g):
            h, w = g.shape
            mid = w // 2
            out = g.copy()
            right = g[:, w - mid:]
            out[:, :mid] = np.fliplr(right)
            return out
        def sym_v_top(g):
            h, w = g.shape
            mid = h // 2
            out = g.copy()
            top = g[:mid, :]
            out[h - mid:, :] = np.flipud(top)
            return out
        def sym_v_bottom(g):
            h, w = g.shape
            mid = h // 2
            out = g.copy()
            bottom = g[h - mid:, :]
            out[:mid, :] = np.flipud(bottom)
            return out

        for fn in (sym_h_left, sym_h_right, sym_v_top, sym_v_bottom):
            try:
                if all(exact(fn(inp), out) for inp, out in train):
                    return fn
            except Exception:
                pass
        return None

    # --- 8. Enclosed Hole Filling ---
    def _try_hole_filling(self, train):
        for fill_c in range(10):
            def make_fn(fc):
                def fn(g):
                    h, w = g.shape
                    out = g.copy()
                    visited = np.zeros((h, w), dtype=bool)
                    stack = []
                    for r in range(h):
                        for c in (0, w-1):
                            if g[r, c] == 0 and not visited[r, c]:
                                visited[r, c] = True
                                stack.append((r, c))
                    for c in range(w):
                        for r in (0, h-1):
                            if g[r, c] == 0 and not visited[r, c]:
                                visited[r, c] = True
                                stack.append((r, c))
                    while stack:
                        r, c = stack.pop()
                        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                            nr, nc = r+dr, c+dc
                            if 0 <= nr < h and 0 <= nc < w:
                                if g[nr, nc] == 0 and not visited[nr, nc]:
                                    visited[nr, nc] = True
                                    stack.append((nr, nc))
                    for r in range(h):
                        for c in range(w):
                            if g[r, c] == 0 and not visited[r, c]:
                                out[r, c] = fc
                    return out
                return fn
            fn = make_fn(fill_c)
            try:
                if all(exact(fn(inp), out) for inp, out in train):
                    return fn
            except Exception:
                pass
        return None

    # --- 9. Gravity & Physics ---
    def _try_gravity(self, train):
        for direction in ("down", "up", "left", "right"):
            def make_grav(d=direction):
                def fn(g):
                    h, w = g.shape
                    out = np.zeros_like(g)
                    if d == "down":
                        for c in range(w):
                            col = g[:, c]
                            nonzeros = col[col != 0]
                            out[h - len(nonzeros):, c] = nonzeros
                    elif d == "up":
                        for c in range(w):
                            col = g[:, c]
                            nonzeros = col[col != 0]
                            out[:len(nonzeros), c] = nonzeros
                    elif d == "right":
                        for r in range(h):
                            row = g[r, :]
                            nonzeros = row[row != 0]
                            out[r, w - len(nonzeros):] = nonzeros
                    elif d == "left":
                        for r in range(h):
                            row = g[r, :]
                            nonzeros = row[row != 0]
                            out[r, :len(nonzeros)] = nonzeros
                    return out
                return fn
            fn = make_grav()
            try:
                if all(exact(fn(inp), out) for inp, out in train):
                    return fn
            except Exception:
                pass

        for bar_c in range(1, 10):
            def make_bar_grav(bc=bar_c):
                def fn(g):
                    h, w = g.shape
                    out = np.zeros_like(g)
                    out[g == bc] = bc
                    for c in range(w):
                        bar_rows = np.where(g[:, c] == bc)[0]
                        non_bar = [r for r in range(h) if g[r, c] != 0 and g[r, c] != bc]
                        if len(bar_rows) > 0 and len(non_bar) > 0:
                            top_bar = bar_rows[0]
                            for i, orig_r in enumerate(reversed(non_bar)):
                                target_r = top_bar - 1 - i
                                if 0 <= target_r < h:
                                    out[target_r, c] = g[orig_r, c]
                        else:
                            for r in non_bar:
                                out[r, c] = g[r, c]
                    return out
                return fn
            fn = make_bar_grav()
            try:
                if all(exact(fn(inp), out) for inp, out in train):
                    return fn
            except Exception:
                pass
        return None

    # --- 10. Lines & Raycasting ---
    def _try_lines_and_rays(self, train):
        def connect_dots(g):
            h, w = g.shape
            out = g.copy()
            for color in np.unique(g):
                if color == 0: continue
                rows, cols = np.where(g == color)
                pts = list(zip(rows, cols))
                for i in range(len(pts)):
                    for j in range(i+1, len(pts)):
                        r1, c1 = pts[i]
                        r2, c2 = pts[j]
                        if r1 == r2:
                            out[r1, min(c1, c2):max(c1, c2)+1] = color
                        elif c1 == c2:
                            out[min(r1, r2):max(r1, r2)+1, c1] = color
            return out
        try:
            if all(exact(connect_dots(inp), out) for inp, out in train):
                return connect_dots
        except Exception:
            pass

        for cross_c in range(10):
            def make_cross(cc=cross_c):
                def fn(g):
                    h, w = g.shape
                    out = g.copy()
                    for r in range(h):
                        for c in range(w):
                            if g[r, c] != 0:
                                col = cc if cc != 0 else g[r, c]
                                out[r, :] = np.where(out[r, :] == 0, col, out[r, :])
                                out[:, c] = np.where(out[:, c] == 0, col, out[:, c])
                    return out
                return fn
            fn = make_cross()
            try:
                if all(exact(fn(inp), out) for inp, out in train):
                    return fn
            except Exception:
                pass
        return None

    # --- 11. Object Extraction & Filtering ---
    def _try_object_filtering(self, train):
        for conn in (4, 8):
            for mono in (True, False):
                for mode in ("largest", "smallest", "most_colors"):
                    def make_extractor(c=conn, m=mono, md=mode):
                        def fn(g):
                            objs = extract_objects(g, connectivity=c, mono=m)
                            if not objs: return g
                            if md == "largest":
                                target = max(objs, key=lambda o: o.area)
                            elif md == "smallest":
                                target = min(objs, key=lambda o: o.area)
                            elif md == "most_colors":
                                target = max(objs, key=lambda o: len(np.unique(o.subgrid)))
                            else:
                                target = objs[0]
                            return target.subgrid
                        return fn
                    fn = make_extractor()
                    try:
                        if all(exact(fn(inp), out) for inp, out in train):
                            return fn
                    except Exception:
                        pass
        return None

    # --- 12. Object Recoloring by Area Rank ---
    def _try_object_recolor(self, train):
        for conn in (4, 8):
            def check_recolor(c=conn):
                inp0, out0 = train[0]
                if inp0.shape != out0.shape: return None
                objs0 = extract_objects(inp0, connectivity=c)
                if len(objs0) < 2: return None
                objs0.sort(key=lambda o: o.area)
                pal = []
                for o in objs0:
                    out_colors = [out0[r, c] for r, c in o.cells]
                    if len(set(out_colors)) != 1: return None
                    pal.append(out_colors[0])
                
                def fn(g):
                    out = g.copy()
                    objs = extract_objects(g, connectivity=c)
                    objs.sort(key=lambda o: o.area)
                    for i, o in enumerate(objs):
                        if i < len(pal):
                            for r, c in o.cells:
                                out[r, c] = pal[i]
                    return out
                return fn
            fn = check_recolor()
            if fn:
                try:
                    if all(exact(fn(inp), out) for inp, out in train):
                        return fn
                except Exception:
                    pass
        return None

    # --- 13. Fill Bounding Box ---
    def _try_bbox_fill(self, train):
        for conn in (4, 8):
            def make_fill(c=conn):
                def fn(g):
                    out = g.copy()
                    objs = extract_objects(g, connectivity=c)
                    for o in objs:
                        min_r, min_c, max_r, max_c = o.bbox
                        out[min_r:max_r+1, min_c:max_c+1] = o.color
                    return out
                return fn
            fn = make_fill()
            try:
                if all(exact(fn(inp), out) for inp, out in train):
                    return fn
            except Exception:
                pass
        return None

    # --- 14. Periodic Pattern Extension ---
    def _try_periodic_extension(self, train):
        for period_h in (2, 3, 4):
            for period_w in (2, 3, 4):
                for target_c in range(10):
                    def make_ext(ph=period_h, pw=period_w, tc=target_c):
                        def fn(g):
                            h, w = g.shape
                            block = g[:ph, :pw]
                            if tc != 0:
                                block = np.where(block != 0, tc, 0)
                            rep_y = (h + ph - 1) // ph + 1
                            rep_x = (w + pw - 1) // pw + 1
                            big = np.tile(block, (rep_y, rep_x))
                            return big[:h+ph, :w]
                        return fn
                    fn = make_ext()
                    try:
                        if all(exact(fn(inp), out) for inp, out in train):
                            return fn
                    except Exception:
                        pass
        return None

    # --- 15. Tiling ---
    def _try_tiling(self, train):
        for ny in (2, 3):
            for nx in (2, 3):
                def make_tile(y=ny, x=nx):
                    return lambda g: np.tile(g, (y, x))
                fn = make_tile()
                try:
                    if all(exact(fn(inp), out) for inp, out in train):
                        return fn
                except Exception:
                    pass
        return None

    # --- 16. Multi-Step Compositions ---
    def _try_composition(self, train):
        for rot in (1, 2, 3):
            def make_crop_rot(r=rot):
                def fn(g):
                    rows, cols = np.where(g != 0)
                    if len(rows) == 0: return g
                    sub = g[rows.min():rows.max()+1, cols.min():cols.max()+1]
                    return np.rot90(sub, r)
                return fn
            fn = make_crop_rot()
            try:
                if all(exact(fn(inp), out) for inp, out in train):
                    return fn
            except Exception:
                pass
        return None


# ============================================================
# BENCHMARK RUNNER
# ============================================================

def run_benchmark(data_dir: str = "arc_data", split: str = "all", limit: int = 0):
    import argparse
    root = Path(data_dir)
    if split == "training":
        tasks = sorted((root / "training").glob("*.json"))
    elif split == "evaluation":
        tasks = sorted((root / "evaluation").glob("*.json"))
    elif split == "all":
        tasks = sorted(root.rglob("*.json"))
    else:
        tasks = sorted(Path(split).glob("*.json")) if Path(split).exists() else sorted(root.glob("*.json"))

    if limit > 0:
        tasks = tasks[:limit]

    print("=" * 80, flush=True)
    print("MATHX GPU-ACCELERATED ARC-AGI-1 BENCHMARK RUNNER", flush=True)
    print("=" * 80, flush=True)
    
    engine = GPUSolverEngine()
    print(f"Dataset Split:             {split.upper()}", flush=True)
    print(f"Total Tasks Loaded:        {len(tasks)} tasks", flush=True)
    print(f"Hardware Compute Device:   {engine.gpu.device_name}\n", flush=True)

    solved_tasks = 0
    test_solved_tasks = 0
    total_test_ex = 0
    correct_test_ex = 0

    start_time = time.perf_counter()

    for idx, fpath in enumerate(tasks, 1):
        task_data = json.loads(fpath.read_text(encoding="utf-8"))
        prog = engine.synthesize(task_data)
        
        c = 0
        t = len(task_data["test"])
        total_test_ex += t

        if prog is not None:
            solved_tasks += 1
            for ex in task_data["test"]:
                pred = prog(G(ex["input"]))
                if exact(pred, G(ex["output"])):
                    c += 1
            correct_test_ex += c
            if c == t:
                test_solved_tasks += 1
                status = "SOLVED"
            else:
                status = f"PARTIAL ({c}/{t})"
        else:
            status = "NO_PROGRAM"

        if idx <= 15 or idx % 50 == 0 or idx == len(tasks):
            print(f"[{idx:03d}/{len(tasks)}] Task {fpath.stem:<10} | {status:<15} | Test: {c}/{t}", flush=True)

    total_time = time.perf_counter() - start_time
    avg_task_ms = (total_time / len(tasks)) * 1000 if tasks else 0

    print("\n" + "=" * 80, flush=True)
    print("FINAL BENCHMARK RESULTS (NVIDIA MX330 GPU)", flush=True)
    print("=" * 80, flush=True)
    print(f"Compute Device:            {engine.gpu.device_name}", flush=True)
    print(f"Total GPU Dispatches:      {engine.gpu.dispatches}", flush=True)
    print(f"Dataset Split:             {split.upper()}", flush=True)
    print(f"Total Tasks Evaluated:     {len(tasks)}", flush=True)
    print(f"Training Fit Tasks:        {solved_tasks}/{len(tasks)} ({100*solved_tasks/len(tasks):.2f}%)", flush=True)
    print(f"Exact Test Tasks Solved:   {test_solved_tasks}/{len(tasks)} ({100*test_solved_tasks/len(tasks):.2f}%)", flush=True)
    print(f"Test Example Accuracy:     {correct_test_ex}/{total_test_ex} ({100*correct_test_ex/total_test_ex:.2f}%)", flush=True)
    print(f"Total Execution Time:      {total_time:.2f} seconds", flush=True)
    print(f"Average Time per Task:     {avg_task_ms:.2f} ms", flush=True)
    print("=" * 80, flush=True)

    # Save report
    report = {
        "device": engine.gpu.device_name,
        "split": split,
        "gpu_dispatches": engine.gpu.dispatches,
        "tasks": len(tasks),
        "training_fit": solved_tasks,
        "exact_test_solved": test_solved_tasks,
        "test_examples_correct": correct_test_ex,
        "test_examples_total": total_test_ex,
        "total_time_seconds": total_time,
        "avg_ms_per_task": avg_task_ms,
    }
    Path("mathx_gpu_benchmark_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Benchmark report saved to: mathx_gpu_benchmark_report.json", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ARC-AGI-1 GPU Benchmark Runner")
    parser.add_argument("--data", default="arc_data", help="Root data directory")
    parser.add_argument("--split", default="all", choices=["all", "training", "evaluation"], help="Dataset split to evaluate")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of tasks to evaluate")
    args = parser.parse_args()
    run_benchmark(data_dir=args.data, split=args.split, limit=args.limit)


if __name__ == "__main__":
    main()

