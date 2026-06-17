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
    """Draw a chart-like basemap: water first, land on top, thin coastlines."""
    if ocean_gdf is not None and not ocean_gdf.empty:
        ocean_gdf.plot(ax=ax, facecolor="#b8dce8", edgecolor="none", zorder=0, label='Ocean')
    if land_gdf is not None and not land_gdf.empty:
        land_gdf.plot(ax=ax, facecolor="#f7f7f5", edgecolor="#c8c8c8", linewidth=0.3, zorder=1, label='Land')
    if water_gdf is not None and not water_gdf.empty:
        water_gdf.plot(ax=ax, facecolor="#b8dce8", edgecolor="#9cc8d3", linewidth=0.25, alpha=0.98, zorder=2, label='Water')
    if coast_gdf is not None and not coast_gdf.empty:
        coast_gdf.plot(ax=ax, color="#8fb8c1", linewidth=0.55, zorder=3, label='Coastline')
    if ferry_route_gdf is not None and not ferry_route_gdf.empty:
        ferry_route_gdf.plot(ax=ax, color="#f39c12", linestyle=(0, (5, 3)), linewidth=1.2, zorder=4, label='Traffic Separation Scheme')

    # Keep the full frame extent (avoid aspect cropping)
    if frame_gdf is not None and not frame_gdf.empty:
        minx, miny, maxx, maxy = frame_gdf.total_bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

    ax.set_autoscale_on(False)
    ax.set_aspect('equal', adjustable='box')
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


# =========================
# AIS-style animated asset helpers
# =========================

def _heading_unit_from_yaw(yaw_rad):
    """SIT yaw convention: yaw=0 points north; plot coordinates are x=east, y=north."""
    return float(np.sin(yaw_rad)), float(np.cos(yaw_rad))


def _ais_polygon_xy(east, north, yaw_rad, length_m, width_m):
    """Return an elongated AIS-like vessel polygon in plot coordinates (east, north)."""
    he, hn = _heading_unit_from_yaw(yaw_rad)
    re, rn = np.cos(yaw_rad), -np.sin(yaw_rad)  # right-hand normal; yaw=0 -> east
    pts = np.array([
        [ 0.60,  0.00],
        [ 0.20,  0.50],
        [-0.48,  0.36],
        [-0.58,  0.00],
        [-0.48, -0.36],
        [ 0.20, -0.50],
    ], dtype=float)
    return np.asarray([
        (east + lo * length_m * he + la * width_m * re,
         north + lo * length_m * hn + la * width_m * rn)
        for lo, la in pts
    ])


def _asset_length(asset, default=80.0):
    return float(getattr(getattr(asset, 'info', None), 'ship_length',
                         getattr(getattr(asset, 'ship_model', None), 'l_ship', default)))


def _ais_symbol_size(ax, asset=None, marker_length_m=None):
    """Visible regional-map symbol size. Deliberately larger than the physical ship."""
    if marker_length_m is not None:
        length = float(marker_length_m)
    else:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        extent_based = 0.013 * min(abs(x1 - x0), abs(y1 - y0))
        physical = _asset_length(asset)
        length = max(5.0 * physical, extent_based)
    return length, 0.38 * length

class MapAnimator:
    """
    Map-only animation with an AIS-like asset display:
    - short dotted track behind each ship,
    - enlarged vessel body at the current position,
    - short heading vector in front of the vessel,
    - status strip for one chosen asset.
    """
    def __init__(self, assets, map_gdfs, interval_ms=500, status_asset_index=0, args=None,
                 asset_display="ais", ais_history_points=18,
                 ais_marker_m=None, ais_vector_m=None, show_full_trajectory=False):
        self.assets = assets
        self.interval = interval_ms
        self.focus_idx = status_asset_index
        self.stop_requested = False
        self.args = args
        self.asset_display = asset_display
        self.ais_history_points = int(ais_history_points)
        self.ais_marker_m = ais_marker_m
        self.ais_vector_m = ais_vector_m
        self.show_full_trajectory = show_full_trajectory

        frame_gdf, ocean_gdf, land_gdf, coast_gdf, water_gdf, ferry_route_gdf = map_gdfs

        def sr(a, k): return np.asarray(a.ship_model.simulation_results[k])
        self.t = sr(self.assets[0], 'time [s]')  # assume aligned

        self.palette = ['#2c7fb8', '#e34a33', '#31a354', '#bdbf00', '#d65db1', '#8c6d31', '#636363', '#17becf', '#756bb1']

        # Cache series per asset
        self.series = []
        for j, a in enumerate(assets):
            s = a.ship_model
            sim = s.simulation_results
            nav = getattr(getattr(s, 'auto_pilot', None), 'navigate', None)
            self.series.append({
                'asset': a,
                'east':  np.asarray(sim['east position [m]'], dtype=float),
                'north': np.asarray(sim['north position [m]'], dtype=float),
                'hdg':   np.radians(np.asarray(sim['yaw angle [deg]'], dtype=float)),
                'label': getattr(getattr(a, 'info', None), 'name_tag', 'Ship'),
                'color': self.palette[j % len(self.palette)],
                'route_e': np.asarray(nav.east, dtype=float) if nav is not None else None,
                'route_n': np.asarray(nav.north, dtype=float) if nav is not None else None,
            })

        # Status arrays for the focus ship
        fm = self.assets[self.focus_idx].ship_model
        self.colav  = np.asarray(getattr(fm, 'colav_active_array', []), dtype=bool)
        self.colli  = np.asarray(getattr(fm, 'collision_array', []), dtype=bool)
        self.navfl  = np.asarray(getattr(fm, 'nav_failure_array', []), dtype=bool)
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

        # Static planned routes and optional faint full tracks
        for s in self.series:
            if s['route_e'] is not None and s['route_n'] is not None:
                self.ax_map.plot(s['route_e'], s['route_n'], linestyle=(0, (5, 3)),
                                 color='#f39c12', lw=1.0, alpha=0.60, zorder=6,
                                 label=f"{s['label']} Route")
                if self.args:
                    for e, n in zip(s['route_e'], s['route_n']):
                        circ = Circle((e, n), radius=self.args.radius_of_acceptance,
                                      fill=True, color='red', alpha=0.12, zorder=5)
                        self.ax_map.add_patch(circ)
            if self.show_full_trajectory:
                self.ax_map.plot(s['east'], s['north'], color=s['color'], lw=0.9,
                                 alpha=0.22, zorder=7, label=f"{s['label']} full track")

        # Pre-compute symbol sizes after map limits are fixed.
        for s in self.series:
            L, W = _ais_symbol_size(self.ax_map, asset=s['asset'], marker_length_m=self.ais_marker_m)
            s['symbol_length'] = L
            s['symbol_width'] = W
            s['vector_length'] = float(self.ais_vector_m) if self.ais_vector_m is not None else 2.4 * L

        # Dynamic artists: short dotted history, heading vector, and vessel polygon.
        self.trails, self.heading_lines, self.ship_icons = [], [], []
        for s in self.series:
            tr, = self.ax_map.plot([], [], color=s['color'], lw=1.45,
                                   linestyle=(0, (3, 3)), alpha=0.95, zorder=10,
                                   label=s['label'])
            vec, = self.ax_map.plot([], [], color='0.12', lw=1.1, alpha=0.85, zorder=11)
            icon = Polygon(np.zeros((6, 2)), closed=True, facecolor=s['color'],
                           edgecolor='black', linewidth=0.60, alpha=0.96, zorder=12,
                           animated=True)
            self.ax_map.add_patch(icon)
            tr.set_animated(True)
            vec.set_animated(True)
            self.trails.append(tr)
            self.heading_lines.append(vec)
            self.ship_icons.append(icon)

        self.ax_map.legend(loc='upper right', fontsize=9, framealpha=0.90)

        self.time_text = self.ax_map.text(0.01, 0.97, '', transform=self.ax_map.transAxes,
                                          fontsize=12, va='top', animated=True,
                                          bbox=dict(boxstyle='round,pad=0.20', fc='white', ec='none', alpha=0.75))

        # Status boxes (bottom)
        self.boxes = []
        self._status_artists = []
        for i, lab in enumerate(["COLAV", "Collision", "Nav Failure", "Grounding"]):
            x0, w = i/4.0, 1/4.0
            rect = Rectangle((x0, 0.05), w-0.02, 0.9, fill=False, lw=1.2)
            rect.set_animated(True)
            self.ax_stat.add_patch(rect)
            txt = self.ax_stat.text(x0 + w/2 - 0.01, 0.5,
                                    f"{lab}\nINACTIVE", ha='center', va='center',
                                    fontsize=10, animated=True)
            self.boxes.append((rect, txt))
            self._status_artists.extend([rect, txt])

        self.ax_stat.set_xlim(0, 1); self.ax_stat.set_ylim(0, 1)
        self.fig.canvas.mpl_connect('key_press_event',
                                    lambda e: self.request_stop() if e.key == 'q' else None)

    def request_stop(self):
        self.stop_requested = True
        try:
            self.animation.event_source.stop()
        except Exception:
            pass

    @staticmethod
    def _on(arr, i):
        """Clamp-safe boolean read for status arrays."""
        if not hasattr(arr, 'size') or arr.size == 0:
            return False
        j = i if i < arr.size else arr.size - 1
        return bool(arr[j])

    def _status(self, idx, on, on_text, off_text):
        rect, txt = self.boxes[idx]
        label = txt.get_text().split('\n')[0]
        txt.set_text(f"{label}\n{on_text if on else off_text}")
        rect.set_fill(on); rect.set_alpha(0.15 if on else 0.0); rect.set_hatch('////' if on else '')
        rect.set_edgecolor('#cc0000' if on else 'black')

    def _nav_status(self, idx, warn_on, fail_on):
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

    def _update_status_strip(self, i):
        self.time_text.set_text(f"Time: {self.t[i]:.1f} s")
        self._status(0, self._on(self.colav, i), "ACTIVE", "INACTIVE")
        self._status(1, self._on(self.colli, i), "YES",    "NO")
        self._nav_status(2, self._on(self.navwarn, i), self._on(self.navfl, i))
        self._status(3, self._on(self.ground, i), "YES",   "NO")

    def init_animation(self):
        for tr in self.trails:
            tr.set_data([], [])
        for vec in self.heading_lines:
            vec.set_data([], [])
        for icon in self.ship_icons:
            icon.set_xy(np.zeros((6, 2)))
        self.time_text.set_text('')
        for rect, txt in self.boxes:
            label = txt.get_text().split('\n')[0]
            txt.set_text(label + "\nINACTIVE")
            rect.set_fill(False); rect.set_alpha(0.0); rect.set_hatch('')
            rect.set_edgecolor('black'); rect.set_linewidth(1.2)
        return (*self.trails, *self.heading_lines, *self.ship_icons, self.time_text, *self._status_artists)

    def animate(self, i):
        i = min(i, len(self.t) - 1)

        if not self.stop_requested:
            for s, tr, vec, icon in zip(self.series, self.trails, self.heading_lines, self.ship_icons):
                if s['east'].size == 0:
                    continue
                idx = min(i, s['east'].size - 1, s['north'].size - 1, s['hdg'].size - 1)
                start = max(0, idx - self.ais_history_points)

                tr.set_data(s['east'][start:idx + 1], s['north'][start:idx + 1])

                east, north, psi = float(s['east'][idx]), float(s['north'][idx]), float(s['hdg'][idx])
                he, hn = _heading_unit_from_yaw(psi)
                L, W = s['symbol_length'], s['symbol_width']
                vec_len = s['vector_length']
                vec.set_data([east, east + he * vec_len], [north, north + hn * vec_len])
                icon.set_xy(_ais_polygon_xy(east, north, psi, L, W))

        self._update_status_strip(i)
        return (*self.trails, *self.heading_lines, *self.ship_icons, self.time_text, *self._status_artists)

    def run(self, fps=None, show=False, repeat=False):
        if fps is not None:
            self.interval = 1000 / fps
        self.animation = FuncAnimation(self.fig, self.animate, frames=len(self.t),
                                       init_func=self.init_animation, blit=True,
                                       interval=self.interval, repeat=repeat,
                                       cache_frame_data=False)
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
