# main.py — moderngl-window 3.x version
# compute hits (r32ui) + resolve into accum (rgba16f) + present + imgui sliders + hot reload
# save frame WITHOUT GUI by capturing inside on_render (stable GL state)
#
# deps:
#   python -m pip install moderngl-window pillow
#
# run:
#   python main.py

import os
import math
import time
import pathlib

import moderngl
import moderngl_window as mglw
from moderngl_window.integrations.imgui import ModernglWindowRenderer

import imgui
from PIL import Image

WG_X, WG_Y = 16, 16  # must match local_size in BOTH compute shaders

# Match your Processing domain
X1, X2 = -1.0, 1.0
Y1, Y2 = -1.0, 1.0
MARGIN_FRAC = 0.05


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class HotReloader:
    def __init__(self, *paths: str):
        self.paths = list(paths)
        self.mtimes = {p: 0.0 for p in self.paths}

    def changed(self) -> bool:
        dirty = False
        for p in self.paths:
            try:
                m = os.path.getmtime(p)
            except FileNotFoundError:
                continue
            if m != self.mtimes[p]:
                self.mtimes[p] = m
                dirty = True
        return dirty


def _timestamp_name(prefix="frame", ext=".png"):
    t = time.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{t}{ext}"


class App(mglw.WindowConfig):
    gl_version = (4, 3)
    title = "Temporal accumulation AA (hits -> accum -> present) — mglw"
    window_size = (2000, 2000)
    aspect_ratio = None
    resizable = True
    vsync = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        print("GL_VERSION:", self.ctx.info.get("GL_VERSION"))
        print("GL_RENDERER:", self.ctx.info.get("GL_RENDERER"))

        # --- ImGui ---
        imgui.create_context()
        self.imgui = ModernglWindowRenderer(self.wnd)

        # Shaders
        self.shader_dir = "shaders"
        self.vert_path = f"{self.shader_dir}/vert.glsl"
        self.present_frag_path = f"{self.shader_dir}/present_accum.frag.glsl"
        self.hits_path = f"{self.shader_dir}/hits.compute.glsl"
        self.resolve_path = f"{self.shader_dir}/resolve.compute.glsl"

        self.reloader = HotReloader(
            self.vert_path,
            self.present_frag_path,
            self.hits_path,
            self.resolve_path,
        )

        # Controls
        self.wrap_mode = 3
        self.show_ui = True

        # Temporal AA controls
        self.alpha = 0.06
        self.exposure = 0.015

        # Optional present shaping (only used if present shader declares these uniforms)
        self.density_scale = 1.0
        self.density_gamma = 0.6

        # Variation params (host-side defaults; GLSL should declare as `uniform float name;`)
        self.p = {
            "pdj_a": 0.1, "pdj_b": 1.9, "pdj_c": -0.8, "pdj_d": -1.2,
            "popcorn_c": 0.1, "popcorn_f": 0.09,
            "waves_b": 0.3, "waves_c": 0.5, "waves_e": 1.0, "waves_f": 1.0,
            "rings_c": 0.9,
            "fan_c": 0.1, "fan_f": 0.4,
            "blob_high": 0.9, "blob_low": 0.1, "blob_waves": 10.0,
        }

        # GPU objects
        self.present_prog = None
        self.present_vao = None
        self.cs_hits = None
        self.cs_resolve = None

        # Cached uniforms
        self.h_time = None
        self.h_res = None
        self.h_x1 = None
        self.h_x2 = None
        self.h_y1 = None
        self.h_y2 = None
        self.h_marg = None

        self.r_res = None
        self.r_alpha = None
        self.r_exposure = None

        # Textures
        self.hits_tex = None     # r32ui
        self.accum_tex = None    # rgba16f
        self._hits_zero = None
        self._accum_zero = None

        # Capture target (scene-only)
        self.capture_tex = None  # rgba8
        self.capture_fbo = None

        # Capture request (deferred to on_render)
        self._capture_requested = False
        self._capture_dir = "saves"
        self._capture_prefix = "frame"

        self._alloc_textures()
        self._alloc_capture_target()
        self._build_shaders_or_keep_old(initial=True)

    # -------------------- lifecycle --------------------
    def on_close(self):
        try:
            self.imgui.release()
        except Exception:
            pass

        try:
            if self.capture_fbo is not None:
                self.capture_fbo.release()
            if self.capture_tex is not None:
                self.capture_tex.release()
        except Exception:
            pass

    def on_resize(self, width: int, height: int):
        self._alloc_textures()
        self._alloc_capture_target()
        self.reset_accum()

    # -------------------- utilities --------------------
    def _set_if_present(self, prog, name, value):
        try:
            prog[name].value = value
            return True
        except KeyError:
            return False

    # -------------------- GPU allocation --------------------
    def _alloc_textures(self):
        w, h = self.wnd.buffer_size

        if self.hits_tex is not None:
            self.hits_tex.release()
        if self.accum_tex is not None:
            self.accum_tex.release()

        # hits: r32ui
        self.hits_tex = self.ctx.texture((w, h), components=1, dtype="u4")
        self.hits_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.hits_tex.repeat_x = self.hits_tex.repeat_y = False

        # accum: rgba16f
        self.accum_tex = self.ctx.texture((w, h), components=4, dtype="f2")
        self.accum_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.accum_tex.repeat_x = self.accum_tex.repeat_y = False

        self._hits_zero = b"\x00\x00\x00\x00" * (w * h)
        self._accum_zero = b"\x00" * (w * h * 4 * 2)

        self.hits_tex.write(self._hits_zero)
        self.accum_tex.write(self._accum_zero)

    def _alloc_capture_target(self):
        w, h = self.wnd.buffer_size

        if self.capture_fbo is not None:
            self.capture_fbo.release()
            self.capture_fbo = None
        if self.capture_tex is not None:
            self.capture_tex.release()
            self.capture_tex = None

        self.capture_tex = self.ctx.texture((w, h), components=3, dtype="f2")
        self.capture_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.capture_tex.repeat_x = self.capture_tex.repeat_y = False
        self.capture_fbo = self.ctx.framebuffer(color_attachments=[self.capture_tex])

    def reset_accum(self):
        if self.accum_tex is not None and self._accum_zero is not None:
            self.accum_tex.write(self._accum_zero)

    # -------------------- shader build / hot reload --------------------
    def _release_shader_objects(self):
        if self.present_vao is not None:
            self.present_vao.release()
            self.present_vao = None
        if self.present_prog is not None:
            self.present_prog.release()
            self.present_prog = None
        if self.cs_hits is not None:
            self.cs_hits.release()
            self.cs_hits = None
        if self.cs_resolve is not None:
            self.cs_resolve.release()
            self.cs_resolve = None

    def _build_shaders_or_keep_old(self, initial: bool = False):
        try:
            new_present_prog = self.ctx.program(
                vertex_shader=read_text(self.vert_path),
                fragment_shader=read_text(self.present_frag_path),
            )
            new_present_vao = self.ctx.vertex_array(new_present_prog, [])
            new_present_vao.vertices = 3

            if "u_accum" in new_present_prog:
                new_present_prog["u_accum"].value = 0
            else:
                raise RuntimeError("present shader must have uniform sampler2D u_accum")

            # optional
            if "u_gamma" in new_present_prog:
                new_present_prog["u_gamma"].value = 2.2
            if "u_display_exposure" in new_present_prog:
                new_present_prog["u_display_exposure"].value = 1.0

            new_cs_hits = self.ctx.compute_shader(read_text(self.hits_path))
            new_cs_resolve = self.ctx.compute_shader(read_text(self.resolve_path))

        except Exception as e:
            tag = "INITIAL BUILD FAILED" if initial else "SHADER RELOAD FAILED"
            print(f"\n[{tag}] {e}\n")
            return

        self._release_shader_objects()
        self.present_prog = new_present_prog
        self.present_vao = new_present_vao
        self.cs_hits = new_cs_hits
        self.cs_resolve = new_cs_resolve

        # cached handles (optional)
        self.h_time = self.cs_hits.get("fGlobalTime", None)
        self.h_res  = self.cs_hits.get("v2Resolution", None)
        self.h_x1   = self.cs_hits.get("u_x1", None)
        self.h_x2   = self.cs_hits.get("u_x2", None)
        self.h_y1   = self.cs_hits.get("u_y1", None)
        self.h_y2   = self.cs_hits.get("u_y2", None)
        self.h_marg = self.cs_hits.get("u_margin_frac", None)

        self.r_res      = self.cs_resolve.get("u_resolution", None)
        self.r_alpha    = self.cs_resolve.get("u_alpha", None)
        self.r_exposure = self.cs_resolve.get("u_exposure", None)

        print("[shader reload OK]")

    # -------------------- present pass helper --------------------
    def _present_scene_to_current_fbo(self):
        self._set_if_present(self.present_prog, "u_density_scale", float(self.density_scale))
        self._set_if_present(self.present_prog, "u_density_gamma", float(self.density_gamma))

        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.accum_tex.use(location=0)
        self.present_vao.render()

    # -------------------- capture (deferred) --------------------
    def request_capture(self, out_dir="saves", prefix="frame"):
        self._capture_requested = True
        self._capture_dir = out_dir
        self._capture_prefix = prefix

    def _do_capture_if_requested(self):
        if not self._capture_requested:
            return

        self._capture_requested = False

        pathlib.Path(self._capture_dir).mkdir(parents=True, exist_ok=True)
        w, h = self.wnd.buffer_size

        # Ensure capture target is valid size
        if self.capture_tex is None or self.capture_fbo is None or self.capture_tex.size != (w, h):
            self._alloc_capture_target()

        # Render scene-only into offscreen, then read
        self.capture_fbo.use()
        self.ctx.viewport = (0, 0, w, h)
        self._present_scene_to_current_fbo()

        # Ensure GPU finished before readback
        self.ctx.finish()

        data = self.capture_fbo.read(components=3, alignment=1)
        img = Image.frombytes("RGB", (w, h), data).transpose(Image.FLIP_TOP_BOTTOM)

        path = os.path.join(self._capture_dir, _timestamp_name(prefix=self._capture_prefix, ext=".png"))
        img.save(path)
        print("Saved (no GUI):", path)

        # Restore window framebuffer binding for rest of frame
        self.wnd.fbo.use()

    # -------------------- input forwarding --------------------
    def on_key_event(self, key, action, modifiers):
        self.imgui.key_event(key, action, modifiers)

        if action != self.wnd.keys.ACTION_PRESS:
            return

        if key == self.wnd.keys.ESCAPE:
            self.wnd.close()
        elif key == self.wnd.keys.R:
            self.reset_accum()
        elif key == self.wnd.keys.S:
            self.request_capture(out_dir="saves", prefix="frame")
        elif key == self.wnd.keys.W:
            self.wrap_mode = (self.wrap_mode + 1) % 4
            print("wrap_mode:", self.wrap_mode)
        elif key == self.wnd.keys.TAB:
            self.show_ui = not self.show_ui

    def on_mouse_position_event(self, x, y, dx, dy):
        self.imgui.mouse_position_event(x, y, dx, dy)

    def on_mouse_press_event(self, x, y, button):
        self.imgui.mouse_press_event(x, y, button)

    def on_mouse_release_event(self, x, y, button):
        self.imgui.mouse_release_event(x, y, button)

    def on_unicode_char_entered(self, char):
        self.imgui.unicode_char_entered(char)

    # -------------------- UI --------------------
    def _ui(self):
        if not self.show_ui:
            return

        imgui.begin("Params", True)

        _, self.alpha = imgui.slider_float("alpha (EMA)", float(self.alpha), 0.005, 0.35)
        _, self.exposure = imgui.slider_float(
            "exposure (resolve)", float(self.exposure), 0.0001, 0.5, format="%.6f"
        )

        imgui.separator()
        _, self.density_scale = imgui.slider_float(
            "density_scale (present)", float(self.density_scale), 0.01, 20.0
        )
        _, self.density_gamma = imgui.slider_float(
            "density_gamma (present)", float(self.density_gamma), 0.05, 2.5
        )

        imgui.separator()
        imgui.text(f"wrap_mode: {self.wrap_mode}  (W cycles)")

        if imgui.button("Reset accum (R)"):
            self.reset_accum()
        imgui.same_line()
        if imgui.button("Save frame (S)"):
            self.request_capture(out_dir="saves", prefix="frame")

        imgui.separator()

        if imgui.collapsing_header("PDJ")[0]:
            _, self.p["pdj_a"] = imgui.slider_float("pdj_a", float(self.p["pdj_a"]), -5.0, 5.0)
            _, self.p["pdj_b"] = imgui.slider_float("pdj_b", float(self.p["pdj_b"]), -5.0, 5.0)
            _, self.p["pdj_c"] = imgui.slider_float("pdj_c", float(self.p["pdj_c"]), -5.0, 5.0)
            _, self.p["pdj_d"] = imgui.slider_float("pdj_d", float(self.p["pdj_d"]), -5.0, 5.0)

        if imgui.collapsing_header("Popcorn")[0]:
            _, self.p["popcorn_c"] = imgui.slider_float("popcorn_c", float(self.p["popcorn_c"]), 0.0, 1.0)
            _, self.p["popcorn_f"] = imgui.slider_float("popcorn_f", float(self.p["popcorn_f"]), 0.0, 1.0)

        if imgui.collapsing_header("Waves")[0]:
            _, self.p["waves_b"] = imgui.slider_float("waves_b", float(self.p["waves_b"]), -2.0, 2.0)
            _, self.p["waves_c"] = imgui.slider_float("waves_c", float(self.p["waves_c"]), 0.05, 3.0)
            _, self.p["waves_e"] = imgui.slider_float("waves_e", float(self.p["waves_e"]), -2.0, 2.0)
            _, self.p["waves_f"] = imgui.slider_float("waves_f", float(self.p["waves_f"]), 0.05, 3.0)

        if imgui.collapsing_header("Rings / Fan / Blob")[0]:
            _, self.p["rings_c"] = imgui.slider_float("rings_c", float(self.p["rings_c"]), 0.01, 3.0)
            _, self.p["fan_c"] = imgui.slider_float("fan_c", float(self.p["fan_c"]), 0.0, 2.0)
            _, self.p["fan_f"] = imgui.slider_float("fan_f", float(self.p["fan_f"]), 0.0, 4.0)
            _, self.p["blob_high"] = imgui.slider_float("blob_high", float(self.p["blob_high"]), 0.0, 2.0)
            _, self.p["blob_low"] = imgui.slider_float("blob_low", float(self.p["blob_low"]), 0.0, 2.0)
            _, self.p["blob_waves"] = imgui.slider_float("blob_waves", float(self.p["blob_waves"]), 0.0, 50.0)

        imgui.end()

    # -------------------- render loop --------------------
    def on_render(self, time_s: float, frame_time: float):
        # Hot reload shaders
        if self.reloader.changed():
            self._build_shaders_or_keep_old()

        w, h = self.wnd.buffer_size
        self.ctx.viewport = (0, 0, w, h)

        # ---- UI update ----
        imgui.new_frame()
        self._ui()
        imgui.render()

        # 1) clear hits
        self.hits_tex.write(self._hits_zero)

        # 2) hits compute uniforms
        if self.h_time is not None:
            self.h_time.value = float(time_s)
        if self.h_res is not None:
            self.h_res.value = (float(w), float(h))

        if self.h_x1 is not None: self.h_x1.value = float(X1)
        if self.h_x2 is not None: self.h_x2.value = float(X2)
        if self.h_y1 is not None: self.h_y1.value = float(Y1)
        if self.h_y2 is not None: self.h_y2.value = float(Y2)
        if self.h_marg is not None: self.h_marg.value = float(MARGIN_FRAC)

        self._set_if_present(self.cs_hits, "u_wrap_mode", int(self.wrap_mode))
        for k, v in self.p.items():
            self._set_if_present(self.cs_hits, k, float(v))

        # dispatch hits
        self.hits_tex.bind_to_image(0, read=False, write=True)
        gx = math.ceil(w / WG_X)
        gy = math.ceil(h / WG_Y)
        self.cs_hits.run(gx, gy, 1)

        # 3) resolve uniforms
        if self.r_res is not None:
            self.r_res.value = (float(w), float(h))
        if self.r_alpha is not None:
            self.r_alpha.value = float(self.alpha)
        if self.r_exposure is not None:
            self.r_exposure.value = float(self.exposure)

        # dispatch resolve
        self.hits_tex.bind_to_image(0, read=True, write=False)
        self.accum_tex.bind_to_image(1, read=True, write=True)
        self.cs_resolve.run(gx, gy, 1)

        # --- capture (scene only) happens HERE, in-frame ---
        self._do_capture_if_requested()

        # 4) present to WINDOW
        self.wnd.fbo.use()
        self.ctx.viewport = (0, 0, w, h)
        self._present_scene_to_current_fbo()

        # 5) draw UI on top
        self.imgui.render(imgui.get_draw_data())


if __name__ == "__main__":
    mglw.run_window_config(App)
