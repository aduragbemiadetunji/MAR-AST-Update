import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Polygon, Rectangle, Circle
from matplotlib import gridspec as mpl_gs
from matplotlib.patches import FancyArrowPatch
from pathlib import Path
# =========================
# Shared helpers
# =========================

def draw_real_map(ax, land_gdf, ocean_gdf, water_gdf, coast_gdf, frame_gdf, ferry_route_gdf):
    # Draw order
    if not land_gdf.empty:
        land_gdf.plot(ax=ax, facecolor="#e6e6e6", edgecolor="#bbbbbb", linewidth=0.3, zorder=0, label='Land')
    if not ocean_gdf.empty:
        ocean_gdf.plot(ax=ax, facecolor="#a0c8f0", edgecolor="none", zorder=1, label='Ocean')
    if not water_gdf.empty:
        water_gdf.plot(ax=ax, facecolor="#a0c8f0", edgecolor="#74a8d8", linewidth=0.4, alpha=0.95, zorder=2, label='Shallow Water')
    if not coast_gdf.empty:
        coast_gdf.plot(ax=ax, color="#a0c8f0", linewidth=1.2, zorder=3, label='Coastline')
    if not ferry_route_gdf.empty:
        ferry_route_gdf.plot(ax=ax, color="#432466", linestyle='--', linewidth=1.2, zorder=3, label='Traffic Separation Scheme')

    # Keep the full frame extent (avoid aspect cropping)
    if frame_gdf is not None and not frame_gdf.empty:
        minx, miny, maxx, maxy = frame_gdf.total_bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

    ax.set_autoscale_on(False)
    ax.set_aspect('equal', adjustable='box')
    # ax.set_anchor('C')
    ax.set_axis_off()
    ax.margins(0)
    ax.grid(False)

def setup_nav_polar(ax, rmax):
    ax.set_rlim(0, max(1.0, rmax))
    ax.set_theta_zero_location('N')   # 0° at North
    ax.set_theta_direction(-1)        # CW positive
    ax.set_thetagrids([0,45,90,135,180,225,270,315],
                      labels=['0','45°','90°','135°','180°','-135°','-90°','-45°'])
    ax.set_rlabel_position(225)
    ax.grid(alpha=0.3)

def _wrap180(a_deg):  # [-180, 180]
    return ((a_deg + 180.0) % 360.0) - 180.0

def _math_to_nav(deg_math):  # 0E/CCW+ -> 0N/CW+
    return _wrap180(90.0 - deg_math)

def ship_vertices_polar(ship_draw, heading_rad_nav, ax, size_frac=0.25):
    """Vertices in (theta, r) for a ShipDraw outline, rotated by heading."""
    x_loc, y_loc = ship_draw.local_coords()
    rmax = ax.get_rmax()
    scale = (size_frac * rmax) / (ship_draw.l / 2.0)
    x = x_loc * scale; y = y_loc * scale
    theta_math = np.arctan2(y, x)
    theta_nav  = theta_math + heading_rad_nav
    r = np.hypot(x, y)
    return np.column_stack([theta_nav, r])

def _make_arrow(ax, lw=2, head=14):
    arr = FancyArrowPatch((0, 0), (0, 0),
                          arrowstyle='-|>',      # triangular head
                          mutation_scale=head,   # head size
                          lw=lw,
                          transform=ax.transData,  # theta,r in DATA coords
                          clip_on=False,
                          animated=True)         # important for blitting
    ax.add_patch(arr)
    return arr

# -------- Window positioning (backend-safe-ish) --------
def get_screen_size(fig):
    m = fig.canvas.manager
    # Tk
    try:
        return m.window.winfo_screenwidth(), m.window.winfo_screenheight()
    except Exception:
        pass
    # Qt
    try:
        scr = m.window.screen() if hasattr(m.window, "screen") else m.window.windowHandle().screen()
        geo = scr.availableGeometry()
        return geo.width(), geo.height()
    except Exception:
        pass
    # Fallback
    return 1920, 1080

def move_figure(fig, x, y, w=None, h=None):
    m = fig.canvas.manager
    backend = plt.get_backend().lower()
    try:
        if 'tk' in backend:
            if w is None or h is None:
                m.window.wm_geometry(f"+{int(x)}+{int(y)}")
            else:
                m.window.geometry(f"{int(w)}x{int(h)}+{int(x)}+{int(y)}")
        elif 'qt' in backend:
            if w is not None and h is not None:
                m.window.setGeometry(int(x), int(y), int(w), int(h))
            else:
                m.window.move(int(x), int(y))
        elif 'wx' in backend:
            m.window.SetPosition((int(x), int(y)))
            if w is not None and h is not None:
                m.window.SetSize((int(w), int(h)))
    except Exception:
        # Last-ditch: ignore failures silently
        pass

def animate_side_by_side(fig_left, fig_right, left_frac=0.68, height_frac=0.92, gap_px=12, show=True):
    """Center both windows on screen, same height, side-by-side."""
    sw, sh = get_screen_size(fig_left)
    H = int(sh * height_frac)
    W_left  = int(sw * left_frac) - gap_px // 2
    W_right = int(sw * (1 - left_frac)) - gap_px // 2
    total_w = W_left + gap_px + W_right
    x0 = (sw - total_w) // 2
    y0 = (sh - H) // 2
    move_figure(fig_left,  x0,              y0, W_left,  H)
    move_figure(fig_right, x0 + W_left + gap_px, y0, W_right, H)
    if show:
        plt.show()

# =========================
# 1) MAP-ONLY ANIMATOR (with status strip)
# =========================

class MapAnimator:
    """
    Map-only animation with basemap + ships + status strip (for one chosen asset).
    - Blit-friendly (status artists returned each frame)
    - Safe index clamping for status arrays
    - 3-state navigation box via _nav_status(): INACTIVE / WARNING / ACTIVE
    """
    def __init__(self, assets, map_gdfs, interval_ms=500, status_asset_index=0, args=None):
        self.assets = assets
        self.interval = interval_ms
        self.focus_idx = status_asset_index
        self.stop_requested = False
        self.args = args

        # NOTE: order here should match your draw_real_map signature
        frame_gdf, ocean_gdf, land_gdf, coast_gdf, water_gdf, ferry_route_gdf = map_gdfs

        def sr(a, k): return np.asarray(a.ship_model.simulation_results[k])
        self.t = sr(self.assets[0], 'time [s]')  # assume aligned

        # Cache series per asset
        self.series = []
        for a in assets:
            s = a.ship_model
            sim = s.simulation_results
            self.series.append({
                'east':  np.asarray(sim['east position [m]']),
                'north': np.asarray(sim['north position [m]']),
                'hdg':   np.radians(np.asarray(sim['yaw angle [deg]'])),
                'label': getattr(getattr(a, 'info', None), 'name_tag', 'Ship'),
                'drawer': getattr(s, 'draw', None),
                'route_e': getattr(getattr(s, 'auto_pilot', None), 'navigate', None).east
                           if hasattr(getattr(s, 'auto_pilot', None), 'navigate') else None,
                'route_n': getattr(getattr(s, 'auto_pilot', None), 'navigate', None).north
                           if hasattr(getattr(s, 'auto_pilot', None), 'navigate') else None,
            })

        # Status arrays for the focus ship
        fm = self.assets[self.focus_idx].ship_model
        self.colav  = np.asarray(getattr(fm, 'colav_active_array', []), dtype=bool)
        self.colli  = np.asarray(getattr(fm, 'collision_array', []), dtype=bool)
        self.navfl  = np.asarray(getattr(fm, 'nav_failure_array', []), dtype=bool)
        # Optional: navigation warning array; default to zeros with same length as failure
        self.navwarn = np.asarray(getattr(fm, 'nav_warning_array', []), dtype=bool)
        if self.navwarn.size == 0 and self.navfl.size:
            self.navwarn = np.zeros_like(self.navfl, dtype=bool)
        self.ground = np.asarray(getattr(fm, 'grounding_array', []), dtype=bool)

        # ---- Figure: map + status strip
        self.fig = plt.figure(figsize=(17.5, 9.8), constrained_layout=True)
        gs = self.fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.0, 0.12], hspace=0.0)
        self.ax_map   = self.fig.add_subplot(gs[0, 0])
        self.ax_stat  = self.fig.add_subplot(gs[1, 0]); self.ax_stat.set_axis_off()

        draw_real_map(self.ax_map, land_gdf, ocean_gdf, water_gdf, coast_gdf, frame_gdf, ferry_route_gdf)
        self.ax_map.set_title("Ship Trajectory")

        # Static routes
        for s in self.series:
            if s['route_e'] is not None and s['route_n'] is not None:
                self.ax_map.plot(s['route_e'], s['route_n'], '--', alpha=0.55, zorder=6, label=f"{s['label']} Route")
                self.ax_map.scatter(s['route_e'], s['route_n'], marker='x', alpha=0.7, zorder=7)

                if self.args:
                    for e, n in zip(s['route_e'], s['route_n']):
                        circ = Circle(
                        (e, n),
                        radius=self.args.radius_of_acceptance,
                        fill=True,
                        color='red',
                        alpha=0.2
                    )
                        self.ax_map.add_patch(circ)

        # Dynamic artists
        self.trails, self.outlines = [], []
        for s in self.series:
            tr, = self.ax_map.plot([], [], '-', lw=1.8, zorder=10, label=s['label'])
            ol, = self.ax_map.plot([], [], lw=1.4, zorder=12)
            tr.set_animated(True); ol.set_animated(True)
            self.trails.append(tr); self.outlines.append(ol)

        self.ax_map.legend(loc='upper right')

        self.time_text = self.ax_map.text(0.01, 0.97, '',
                                          transform=self.ax_map.transAxes,
                                          fontsize=12, va='top', animated=True)

        # Status boxes (bottom) — animated + collected
        self.boxes = []
        self._status_artists = []
        for i, lab in enumerate(["COLAV", "Collision", "Nav Failure", "Grounding"]):
            x0, w = i/4.0, 1/4.0
            rect = Rectangle((x0, 0.05), w-0.02, 0.9, fill=False, lw=1.2)
            rect.set_animated(True)
            self.ax_stat.add_patch(rect)
            txt = self.ax_stat.text(x0 + w/2 - 0.01, 0.5,
                                    f"{lab}\nINACTIVE",
                                    ha='center', va='center',
                                    fontsize=10, animated=True)
            self.boxes.append((rect, txt))
            self._status_artists.extend([rect, txt])

        self.ax_stat.set_xlim(0, 1); self.ax_stat.set_ylim(0, 1)

        # 'q' to stop
        self.fig.canvas.mpl_connect('key_press_event',
                                    lambda e: self.request_stop() if e.key == 'q' else None)

    def request_stop(self):
        self.stop_requested = True
        try:
            self.animation.event_source.stop()
        except Exception:
            pass

    # ---------- helpers ----------
    @staticmethod
    def _on(arr, i):
        """Clamp-safe boolean read for status arrays."""
        if not hasattr(arr, 'size') or arr.size == 0:
            return False
        j = i if i < arr.size else arr.size - 1
        return bool(arr[j])

    def _status(self, idx, on, on_text, off_text):
        """Generic 2-state renderer."""
        rect, txt = self.boxes[idx]
        label = txt.get_text().split('\n')[0]
        txt.set_text(f"{label}\n{on_text if on else off_text}")
        rect.set_fill(on); rect.set_alpha(0.15 if on else 0.0); rect.set_hatch('////' if on else '')
        rect.set_edgecolor('#cc0000' if on else 'black')

    def _nav_status(self, idx, warn_on, fail_on):
        """3-state renderer for navigation box: INACTIVE / WARNING / ACTIVE."""
        rect, txt = self.boxes[idx]
        label = txt.get_text().split('\n')[0]
        if fail_on:
            txt.set_text(f"{label}\nYES")
            rect.set_fill(True);  rect.set_alpha(0.18)
            rect.set_hatch('/////')
            rect.set_edgecolor('#cc0000'); rect.set_linewidth(1.4)
        elif warn_on:
            txt.set_text(f"{label}\nWARNING")
            rect.set_fill(True);  rect.set_alpha(0.12)
            rect.set_hatch('....')
            rect.set_edgecolor("#fffb00"); rect.set_linewidth(1.4)
        else:
            txt.set_text(f"{label}\nNO")
            rect.set_fill(False); rect.set_alpha(0.0)
            rect.set_hatch('')
            rect.set_edgecolor('black'); rect.set_linewidth(1.2)

    # ---------- animation hooks ----------
    def init_animation(self):
        for tr in self.trails: tr.set_data([], [])
        for ol in self.outlines: ol.set_data([], [])
        self.time_text.set_text('')
        # reset boxes to INACTIVE
        for k, (rect, txt) in enumerate(self.boxes):
            label = txt.get_text().split('\n')[0]
            txt.set_text(label + "\nINACTIVE")
            rect.set_fill(False); rect.set_alpha(0.0); rect.set_hatch('')
            rect.set_edgecolor('black'); rect.set_linewidth(1.2)
        return (*self.trails, *self.outlines, self.time_text, *self._status_artists)

    def animate(self, i):
        if self.stop_requested:
            self.time_text.set_text(f"Time: {self.t[i]:.1f} s")
            self._status(0, self._on(self.colav, i), "ACTIVE", "INACTIVE")
            self._status(1, self._on(self.colli, i), "YES",    "NO")
            self._nav_status(2, self._on(self.navwarn, i), self._on(self.navfl, i))
            self._status(3, self._on(self.ground, i), "YES",   "NO")
            return (*self.trails, *self.outlines, self.time_text, *self._status_artists)

        # trails + current outlines
        for s, tr, ol in zip(self.series, self.trails, self.outlines):
            tr.set_data(s['east'][:i], s['north'][:i])
            if s['drawer'] is not None and i < len(s['east']):
                east, north, psi = s['east'][i], s['north'][i], s['hdg'][i]
                xL, yL = s['drawer'].local_coords()
                xR, yR = s['drawer'].rotate_coords(xL, yL, psi)
                xF, yF = s['drawer'].translate_coords(xR, yR, north, east)
                ol.set_data(yF, xF)  # (East, North)

        # time + statuses for focus ship
        self.time_text.set_text(f"Time: {self.t[i]:.1f} s")
        self._status(0, self._on(self.colav, i), "ACTIVE", "INACTIVE")
        self._status(1, self._on(self.colli, i), "YES",    "NO")
        self._nav_status(2, self._on(self.navwarn, i), self._on(self.navfl, i))
        self._status(3, self._on(self.ground, i), "YES",   "NO")

        return (*self.trails, *self.outlines, self.time_text, *self._status_artists)

    def run(self, fps=None, show=False, repeat=False):
        if fps is not None: self.interval = 1000 / fps
        self.animation = FuncAnimation(self.fig, self.animate, frames=len(self.t),
                                       init_func=self.init_animation, blit=True,
                                       interval=self.interval, repeat=repeat)
        if show:
            plt.show()
    
    def save(self, base_path, filename, fps=25):
        self.animation = FuncAnimation(self.fig, self.animate,
                                       frames=len(self.t),
                                       init_func=self.init_animation,
                                       blit=True, interval=self.interval,
                                       repeat=False, cache_frame_data=False)
        base_path = Path(base_path)
        path = base_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        FFMpegWriter(fps=fps).setup(self.fig, path, dpi=140)
        writer = FFMpegWriter(fps=fps)
        self.animation.save(path, writer=writer)

class PolarAnimator:
    """Three polar panels (wave/current/wind) for one focus ship.
       Per-axis rmax; arrows for vectors; blue rotating ship outline."""
    def __init__(self, focus_asset, interval_ms=500, rpad=0.05, rmin=1.0):
        self.a = focus_asset
        self.interval = interval_ms

        sim = self.a.ship_model.simulation_results
        self.t        = np.asarray(sim['time [s]'])
        self.hdg_rad  = np.radians(np.asarray(sim['yaw angle [deg]'])) - np.pi/2.0

        # wave dir from forces (math -> nav)
        dy = np.asarray(sim['wave force north [N]'])
        dx = np.asarray(sim['wave force east [N]'])
        self.wave_mag   = np.sqrt(dx**2 + dy**2) / 1000

        # current & wind (assumed already 0N/CW+)
        self.curr_dir_d = np.asarray(sim['current dir [deg]'])
        self.curr_spd   = np.asarray(sim['current speed [m/s]'])
        self.wind_dir_d = np.asarray(sim['wind dir [deg]'])
        self.wind_spd   = np.asarray(sim['wind speed [m/s]'])

        def _rmax(arr):
            if hasattr(arr, "size") and arr.size:
                return max(rmin, float(np.nanmax(arr)) * (1.0 + rpad))
            return rmin

        self.rmax_wave = _rmax(self.wave_mag)
        self.rmax_curr = _rmax(self.curr_spd)
        self.rmax_wind = _rmax(self.wind_spd)

        # ---- figure (manual spacing) ----
        self.fig = plt.figure(figsize=(6.6, 9.4))
        self.fig.subplots_adjust(left=0.10, right=0.90, top=0.925, bottom=0.075, hspace=0.5)
        self.ax_wave = self.fig.add_subplot(311, projection='polar')
        self.ax_curr = self.fig.add_subplot(312, projection='polar')
        self.ax_wind = self.fig.add_subplot(313, projection='polar')

        def _setup(ax, rmax, title):
            ax.set_rlim(0, rmax)
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_thetagrids([0,45,90,135,180,225,270,315],
                              labels=['0°','45°','90°','135°','180°','-135°','-90°','-45°'])
            ax.set_rlabel_position(225)
            ax.grid(alpha=0.25)
            ax.set_title(title, pad=10, fontsize=10)
            # smaller tick labels to avoid clutter
            for lab in ax.get_xticklabels() + ax.get_yticklabels():
                lab.set_fontsize(9)

        _setup(self.ax_wave, self.rmax_wave, "Wave load [kN] & heading [deg]")
        _setup(self.ax_curr, self.rmax_curr, "Current speed [m/s] & heading [deg]")
        _setup(self.ax_wind, self.rmax_wind, "Wind speed [m/s] & heading [deg]")

        # ---- arrows (distinct colors) ----
        def _make_arrow(ax, color, lw=2, head=12):
            arr = FancyArrowPatch((0, 0), (0, 0),
                                  arrowstyle='-|>',
                                  mutation_scale=head,
                                  lw=lw, color=color,
                                  transform=ax.transData,
                                  clip_on=False, animated=True, zorder=6)
            ax.add_patch(arr)
            return arr

        self.wave_arrow = _make_arrow(self.ax_wave,  "#ff0000")
        self.curr_arrow = _make_arrow(self.ax_curr,  "#b095ff")
        self.wind_arrow = _make_arrow(self.ax_wind,  "#56cd32")

        # ---- ship outlines (blue) ----
        def _make_ship(ax):
            p = Polygon([[0, 0]], closed=True, fill=False, lw=1.8,
                        ec='black', clip_on=False, animated=True, zorder=7)
            ax.add_patch(p)
            return p

        self.ship_icon_wave = _make_ship(self.ax_wave)
        self.ship_icon_curr = _make_ship(self.ax_curr)
        self.ship_icon_wind = _make_ship(self.ax_wind)

        # ---- bottom-left time label in its own (tiny) axes; blit-safe ----
        # Align left edge with the subplots, put near the bottom margin
        left = self.ax_wind.get_position().x0
        width = self.ax_wind.get_position().width
        self.ax_time = self.fig.add_axes([left, 0.02, width, 0.03])  # [L, B, W, H] in fig coords
        self.ax_time.set_axis_off()
        self.time_text = self.ax_time.text(0.25, 0.5, '', ha='left', va='center',
                                           fontsize=10, animated=True)

        # artists for blitting
        self._artists = (self.wave_arrow, self.curr_arrow, self.wind_arrow,
                         self.ship_icon_wave, self.ship_icon_curr, self.ship_icon_wind,
                         self.time_text)

        self.ship_draw = getattr(self.a.ship_model, 'draw', None) or self.a.ship_model  # ShipDraw

    # --- helpers ---
    def _ship_verts(self, ax, heading_rad_nav, size_frac=0.24):
        x_loc, y_loc = self.ship_draw.local_coords()
        rmax = ax.get_rmax()
        scale = (size_frac * rmax) / (self.ship_draw.l / 2.0)
        x = x_loc * scale; y = y_loc * scale
        theta_math = np.arctan2(y, x)
        theta_nav  = (np.pi/2.0) - theta_math + heading_rad_nav
        r = np.hypot(x, y)
        return np.column_stack([theta_nav, r])

    # --- FuncAnimation hooks ---
    def init_animation(self):
        self.wave_arrow.set_positions((0.0, 0.0), (0.0, 0.0))
        self.curr_arrow.set_positions((0.0, 0.0), (0.0, 0.0))
        self.wind_arrow.set_positions((0.0, 0.0), (0.0, 0.0))
        self.ship_icon_wave.set_xy([[0, 0]])
        self.ship_icon_curr.set_xy([[0, 0]])
        self.ship_icon_wind.set_xy([[0, 0]])
        self.time_text.set_text('')
        return self._artists

    def animate(self, i):
        hdg = self.hdg_rad[i]

        if self.wave_mag.size and self.wind_dir_d.size:
            th = np.deg2rad(self.wind_dir_d[i]); r = float(self.wave_mag[i])
            self.wave_arrow.set_positions((th, 0.0), (th, r))
        if self.curr_spd.size and self.curr_dir_d.size:
            th = np.deg2rad(self.curr_dir_d[i]); r = float(self.curr_spd[i])
            self.curr_arrow.set_positions((th, 0.0), (th, r))
        if self.wind_spd.size and self.wind_dir_d.size:
            th = np.deg2rad(self.wind_dir_d[i]); r = float(self.wind_spd[i])
            self.wind_arrow.set_positions((th, 0.0), (th, r))

        self.ship_icon_wave.set_xy(self._ship_verts(self.ax_wave, hdg))
        self.ship_icon_curr.set_xy(self._ship_verts(self.ax_curr, hdg))
        self.ship_icon_wind.set_xy(self._ship_verts(self.ax_wind, hdg))

        self.time_text.set_text(f"t = {self.t[i]:.1f} s")
        return self._artists

    def run(self, fps=None, show=False, repeat=False):
        if fps is not None: self.interval = 1000 / fps
        self.animation = FuncAnimation(self.fig, self.animate,
                                       frames=len(self.t),
                                       init_func=self.init_animation,
                                       blit=True, interval=self.interval,
                                       repeat=repeat, cache_frame_data=False)
        if show: plt.show()

    def save(self, base_path, filename, fps=25):
        self.animation = FuncAnimation(self.fig, self.animate,
                                       frames=len(self.t),
                                       init_func=self.init_animation,
                                       blit=True, interval=self.interval,
                                       repeat=False, cache_frame_data=False)
        base_path = Path(base_path)
        path = base_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        FFMpegWriter(fps=fps).setup(self.fig, path, dpi=140)
        writer = FFMpegWriter(fps=fps)
        self.animation.save(path, writer=writer)
