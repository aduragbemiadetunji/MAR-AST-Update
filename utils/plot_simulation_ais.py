import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42    # Use TrueType instead of Type 3
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial' #'DejaVu Sans'  # Or: 'Arial', 'Helvetica'
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon

def wrap_angle_deg(x):
    # wrap to (-pi, pi]
    return (x + 180) % (360) - 180

def center_plot_window():
    try:
        manager = plt.get_current_fig_manager()
        backend = plt.get_backend().lower()

        if "tkagg" in backend:
            # For TkAgg (Tkinter)
            manager.window.update_idletasks()
            screen_width = manager.window.winfo_screenwidth()
            screen_height = manager.window.winfo_screenheight()
            window_width = manager.window.winfo_width()
            window_height = manager.window.winfo_height()
            pos_x = int((screen_width - window_width) / 2)
            pos_y = int((screen_height - window_height) / 2)
            manager.window.geometry(f"+{pos_x}+{pos_y}")

        elif "qt" in backend:
            # For QtAgg, Qt5Agg, qtagg, etc.
            screen = manager.window.screen().availableGeometry()
            screen_width, screen_height = screen.width(), screen.height()
            window_width = manager.window.width()
            window_height = manager.window.height()
            pos_x = int((screen_width - window_width) / 2)
            pos_y = int((screen_height - window_height) / 2)
            manager.window.move(pos_x, pos_y)

        else:
            print(f"Centering not supported for backend: {backend}")

    except Exception as e:
        print("Could not reposition the plot window:", e)




def _series_if_exists(df, col, default=None):
    if col in df.columns:
        return df[col]
    return default

def _mode_to_code(series):
    mode_map = {
        "PTO": 0,
        "MEC": 1,
        "PTI": 2,
        "PTI_1DG": 3,
        "MEC_BLACKOUT": 4,
        "DP_SAFE": 5,
    }
    return series.map(mode_map)

def _shade_fault_windows(ax, t, df, color='tab:red', alpha=0.12):
    if 'fault active' not in df.columns:
        return

    fault = df['fault active'].astype(bool).to_numpy()
    tt = np.asarray(t)

    in_fault = False
    start_idx = None

    for i, active in enumerate(fault):
        if active and not in_fault:
            in_fault = True
            start_idx = i
        elif not active and in_fault:
            ax.axvspan(tt[start_idx], tt[i], color=color, alpha=alpha, lw=0)
            in_fault = False
            start_idx = None

    if in_fault and start_idx is not None:
        ax.axvspan(tt[start_idx], tt[-1], color=color, alpha=alpha, lw=0)

# def _annotate_mode_switches(ax, t, df, y=None):
#     if 'operating mode' not in df.columns:
#         return

#     modes = df['operating mode'].astype(str).to_numpy()
#     tt = np.asarray(t)

#     switch_idx = np.where(modes[1:] != modes[:-1])[0] + 1
#     if len(switch_idx) == 0:
#         return

#     if y is None:
#         ymin, ymax = ax.get_ylim()
#         y = ymax - 0.08 * (ymax - ymin)

#     for idx in switch_idx:
#         ax.axvline(tt[idx], color='0.35', linestyle=':', linewidth=1.0)
#         ax.text(
#             tt[idx], y, modes[idx],
#             rotation=90, va='top', ha='right',
#             fontsize=8, color='0.25',
#             bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.75)
#         )



def _annotate_mode_switches(ax, t, df, y=None, min_label_gap_s=1500.0):
    if 'operating mode' not in df.columns:
        return

    modes = df['operating mode'].astype(str).to_numpy()
    tt = np.asarray(t)

    switch_idx = np.where(modes[1:] != modes[:-1])[0] + 1
    if len(switch_idx) == 0:
        return

    if y is None:
        ymin, ymax = ax.get_ylim()
        y = ymax - 0.08 * (ymax - ymin)

    last_labeled_t = -np.inf
    for idx in switch_idx:
        ax.axvline(tt[idx], color='0.35', linestyle=':', linewidth=1.0)

        if tt[idx] - last_labeled_t >= min_label_gap_s or modes[idx] == 'DP_SAFE':
            ax.text(
                tt[idx], y, modes[idx],
                rotation=90, va='top', ha='right',
                fontsize=8, color='0.25',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.75)
            )
            last_labeled_t = tt[idx]

def _plot_binary_step(ax, t, y, label, color=None):
    ax.step(t, y, where='post', label=label, linewidth=1.3, color=color)



def _has_faults(result_df):
    return ("fault active" in result_df.columns) and result_df["fault active"].astype(bool).any()


def _has_mode_switches(result_df):
    return ("operating mode" in result_df.columns) and (result_df["operating mode"].astype(str).nunique() > 1)


def _maybe_fault_and_mode(ax, t, result_df, include_fault_overlays=True, include_mode_annotations=False):
    if include_fault_overlays and _has_faults(result_df):
        _shade_fault_windows(ax, t, result_df)
    if include_mode_annotations and _has_mode_switches(result_df):
        _annotate_mode_switches(ax, t, result_df)


def _add_scalebar(ax, length_m=None, loc=(0.06, 0.06), linewidth=2.0, text_offset=0.015):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    width = abs(x1 - x0)
    height = abs(y1 - y0)

    if length_m is None:
        target = 0.18 * width
        choices = np.array([200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000], dtype=float)
        length_m = choices[np.argmin(np.abs(choices - target))]

    xs = x0 + loc[0] * width
    ys = y0 + loc[1] * height
    xe = xs + length_m

    ax.plot([xs, xe], [ys, ys], color='k', lw=linewidth, solid_capstyle='butt', zorder=20)
    tick_h = 0.01 * height
    ax.plot([xs, xs], [ys - tick_h, ys + tick_h], color='k', lw=linewidth, zorder=20)
    ax.plot([xe, xe], [ys - tick_h, ys + tick_h], color='k', lw=linewidth, zorder=20)

    label = f"{length_m/1000:.0f} km" if length_m >= 1000 else f"{length_m:.0f} m"
    ax.text((xs + xe) / 2, ys + text_offset * height, label, ha='center', va='bottom', fontsize=9, color='k',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8), zorder=21)
    

# =========================
# AIS-style vessel display helpers
# =========================

def _finite_last_index(e, n):
    """Return the last index where both east/north are finite."""
    e = np.asarray(e, dtype=float)
    n = np.asarray(n, dtype=float)
    valid = np.where(np.isfinite(e) & np.isfinite(n))[0]
    return int(valid[-1]) if valid.size else None


def _heading_unit_from_yaw(yaw_rad):
    """
    SIT convention used in the simulator: yaw=0 points north.
    For plotting x=east, y=north, the heading vector is (sin(yaw), cos(yaw)).
    """
    return float(np.sin(yaw_rad)), float(np.cos(yaw_rad))


def _infer_yaw_rad(e, n, idx, yaw_deg=None):
    """Use yaw column if available; otherwise estimate heading from the last motion segment."""
    if yaw_deg is not None:
        y = np.asarray(yaw_deg, dtype=float)
        if idx < y.size and np.isfinite(y[idx]):
            return np.deg2rad(y[idx])

    e = np.asarray(e, dtype=float)
    n = np.asarray(n, dtype=float)
    j0 = max(0, idx - 5)
    for j in range(idx, j0, -1):
        de = e[j] - e[j - 1]
        dn = n[j] - n[j - 1]
        if np.isfinite(de) and np.isfinite(dn) and (de * de + dn * dn) > 1e-6:
            # inverse of de=sin(yaw), dn=cos(yaw)
            return float(np.arctan2(de, dn))
    return 0.0


def _ais_symbol_size(ax, asset=None, marker_length_m=None):
    """
    AIS symbols must be visible on regional plots, so the symbol is deliberately
    larger than the physical ship length. If marker_length_m is provided, it is used directly.
    """
    if marker_length_m is not None:
        length = float(marker_length_m)
    else:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        extent_based = 0.013 * min(abs(x1 - x0), abs(y1 - y0))
        physical_length = 80.0
        if asset is not None:
            physical_length = float(getattr(getattr(asset, 'info', None), 'ship_length',
                                            getattr(getattr(asset, 'ship_model', None), 'l_ship', physical_length)))
        length = max(5.0 * physical_length, extent_based)
    width = 0.38 * length
    return length, width


def _ais_polygon_xy(east, north, yaw_rad, length_m, width_m):
    """Return an elongated AIS-like vessel polygon in plot coordinates (east, north)."""
    he, hn = _heading_unit_from_yaw(yaw_rad)
    # Right-hand normal in the east/north plot plane. For yaw=0, right points east.
    re, rn = np.cos(yaw_rad), -np.sin(yaw_rad)

    # (longitudinal, lateral) points. Bow is sharp, stern is blunt.
    pts = np.array([
        [ 0.60,  0.00],
        [ 0.20,  0.50],
        [-0.48,  0.36],
        [-0.58,  0.00],
        [-0.48, -0.36],
        [ 0.20, -0.50],
    ], dtype=float)

    verts = []
    for longitudinal, lateral in pts:
        ee = east + longitudinal * length_m * he + lateral * width_m * re
        nn = north + longitudinal * length_m * hn + lateral * width_m * rn
        verts.append((ee, nn))
    return np.asarray(verts)


def _draw_ais_asset(ax, e, n, yaw_rad, color, label=None, asset=None,
                    history_points=18, vector_m=None, marker_length_m=None,
                    zorder=10, annotate=False):
    """Draw one AIS-like asset: short dotted past track, vessel body, and heading vector."""
    idx = _finite_last_index(e, n)
    if idx is None:
        return []

    e = np.asarray(e, dtype=float)
    n = np.asarray(n, dtype=float)
    length_m, width_m = _ais_symbol_size(ax, asset=asset, marker_length_m=marker_length_m)
    if vector_m is None:
        vector_m = 2.4 * length_m

    start = max(0, idx - int(history_points))
    artists = []
    if idx > start:
        trail, = ax.plot(
            e[start:idx + 1], n[start:idx + 1],
            color=color, lw=1.35, linestyle=(0, (3.0, 3.0)), alpha=0.95,
            solid_capstyle='round', zorder=zorder
        )
        artists.append(trail)

    east, north = float(e[idx]), float(n[idx])
    he, hn = _heading_unit_from_yaw(yaw_rad)
    vec, = ax.plot(
        [east, east + he * vector_m], [north, north + hn * vector_m],
        color='0.12', lw=1.1, alpha=0.85, zorder=zorder + 1
    )
    artists.append(vec)

    poly = MplPolygon(
        _ais_polygon_xy(east, north, yaw_rad, length_m, width_m),
        closed=True, facecolor=color, edgecolor='black', linewidth=0.55,
        alpha=0.96, zorder=zorder + 2
    )
    ax.add_patch(poly)
    artists.append(poly)

    if annotate and label:
        txt = ax.text(
            east + 0.45 * length_m, north + 0.45 * length_m, str(label),
            fontsize=8, color='0.15', ha='left', va='bottom', zorder=zorder + 3,
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.70)
        )
        artists.append(txt)

    return artists



def _has_faults(result_df):
    return ("fault active" in result_df.columns) and result_df["fault active"].astype(bool).any()

def _has_mode_switches(result_df):
    return ("operating mode" in result_df.columns) and (result_df["operating mode"].astype(str).nunique() > 1)

def _maybe_fault_and_mode(ax, t, result_df, include_fault_overlays=True, include_mode_annotations=False):
    if include_fault_overlays and _has_faults(result_df):
        _shade_fault_windows(ax, t, result_df)
    if include_mode_annotations and _has_mode_switches(result_df):
        _annotate_mode_switches(ax, t, result_df)


def plot_ship_status(asset, result_df, plot_env_load=False, show=False, profile="paper_baseline"):
    def _ax_style(ax, *, xlabel=False, ylabel=None, xlim_left=0, ypct_series=None):
        ax.grid(color='0.88', linestyle='-', linewidth=0.6)
        ax.set_xlim(left=xlim_left)
        if ylabel:
            ax.set_ylabel(ylabel)
        if xlabel:
            ax.set_xlabel('Time (s)')
        if ypct_series is not None:
            lo, hi = np.nanpercentile(ypct_series, [1, 99])
            if np.isfinite(lo) and np.isfinite(hi) and lo != hi:
                pad = 0.05 * (hi - lo)
                ax.set_ylim(lo - pad, hi + pad)

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.3,
    })

    t = result_df['time [s]'].to_numpy()
    nm = str(asset.info.name_tag)
    has_faults = _has_faults(result_df)
    has_switches = _has_mode_switches(result_df)

    if profile in ("paper", "paper_baseline"):
        include_fault_overlays = has_faults
        include_mode_annotations = False
        fig_1, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6.8), sharex=True, constrained_layout=True)
        axes = axes.flatten()

        speed = np.sqrt(result_df['forward speed [m/s]']**2 + result_df['sideways speed [m/s]']**2)
        axes[0].plot(t, speed, label='Actual')
        axes[0].axhline(y=asset.ship_model.desired_speed, color='red', linestyle='--', linewidth=1.2, label='Desired')
        axes[0].set_title('Speed')
        _ax_style(axes[0], ylabel='Speed (m/s)')
        _maybe_fault_and_mode(axes[0], t, result_df, include_fault_overlays, include_mode_annotations)
        axes[0].legend(loc='upper right', frameon=False, fontsize=8)

        axes[1].plot(t, result_df['cross track error [m]'])
        axes[1].axhline(y=0.0, color='red', linestyle='--', linewidth=1.1)
        if hasattr(asset.ship_model, 'cross_track_error_tolerance'):
            tol = asset.ship_model.cross_track_error_tolerance
            axes[1].axhline(y=tol, color='0.45', linestyle=':', linewidth=0.9)
            axes[1].axhline(y=-tol, color='0.45', linestyle=':', linewidth=0.9)
        axes[1].set_title('Cross-track error')
        _ax_style(axes[1], ylabel='Error (m)')
        _maybe_fault_and_mode(axes[1], t, result_df, include_fault_overlays, include_mode_annotations)

        yaw = wrap_angle_deg(result_df['yaw angle [deg]'])
        axes[2].plot(t, yaw)
        axes[2].axhline(y=0.0, color='red', linestyle='--', linewidth=1.1)
        axes[2].set_title('Yaw')
        _ax_style(axes[2], ylabel='Yaw (deg)')
        axes[2].set_ylim(-180.0, 180.0)
        _maybe_fault_and_mode(axes[2], t, result_df, include_fault_overlays, include_mode_annotations)

        rud_lim = np.rad2deg(asset.ship_model.ship_machinery_model.rudder_ang_max)
        axes[3].plot(t, result_df['rudder angle [deg]'])
        axes[3].axhline(y=0.0, color='red', linestyle='--', linewidth=1.1)
        axes[3].axhline(y=rud_lim, color='0.45', linestyle=':', linewidth=0.9)
        axes[3].axhline(y=-rud_lim, color='0.45', linestyle=':', linewidth=0.9)
        axes[3].set_title('Rudder')
        _ax_style(axes[3], xlabel=True, ylabel='Rudder (deg)')
        axes[3].set_ylim(-rud_lim, rud_lim)
        _maybe_fault_and_mode(axes[3], t, result_df, include_fault_overlays, include_mode_annotations)

        axp = axes[4]
        mech_power = _series_if_exists(result_df, 'power me [kw]')
        mech_avail = _series_if_exists(result_df, 'available power me [kw]')
        elec_power = _series_if_exists(result_df, 'power electrical [kw]')
        elec_avail = _series_if_exists(result_df, 'available power electrical [kw]')
        total_power = _series_if_exists(result_df, 'power [kw]')
        if mech_power is not None:
            axp.plot(t, mech_power, label='Mech. demand')
        if mech_avail is not None:
            axp.plot(t, mech_avail, '--', label='Mech. avail.')
        if elec_power is not None and np.nanmax(np.abs(np.asarray(elec_power))) > 0:
            axp.plot(t, elec_power, label='Elec. demand')
        if elec_avail is not None and np.nanmax(np.abs(np.asarray(elec_avail))) > 0:
            axp.plot(t, elec_avail, '--', label='Elec. avail.')
        if total_power is not None and np.nanmax(np.abs(np.asarray(total_power))) > 0:
            axp.plot(t, total_power, linewidth=1.4, alpha=0.85, label='Total')
        axp.set_title('Power')
        _ax_style(axp, xlabel=True, ylabel='Power (kW)')
        _maybe_fault_and_mode(axp, t, result_df, include_fault_overlays, include_mode_annotations)
        axp.legend(loc='best', frameon=False, fontsize=8)

        axes[5].plot(t, result_df['fuel consumption [kg]'])
        axes[5].set_title('Fuel')
        _ax_style(axes[5], xlabel=True, ylabel='Fuel (kg)')
        _maybe_fault_and_mode(axes[5], t, result_df, include_fault_overlays, include_mode_annotations)

        # Optional extra figures only when informative
        if has_switches and profile == "paper_fault":
            fig_m, axm = plt.subplots(figsize=(12, 2.8), constrained_layout=True)
            mode_code = result_df['operating mode code'] if 'operating mode code' in result_df.columns else _mode_to_code(result_df['operating mode'])
            axm.step(t, mode_code, where='post', linewidth=1.4)
            axm.set_yticks([0, 1, 2, 3, 4, 5])
            axm.set_yticklabels(['PTO', 'MEC', 'PTI', 'PTI_1DG', 'MEC_BLK', 'DP_SAFE'])
            axm.set_title('Machinery mode timeline')
            _ax_style(axm, xlabel=True, ylabel='Mode')
            _maybe_fault_and_mode(axm, t, result_df, include_fault_overlays, False)

        if has_faults and {'ME available', 'DG1 available', 'DG2 available', 'HSG available'}.issubset(result_df.columns):
            fig_eng, ax_eng = plt.subplots(figsize=(12, 2.8), constrained_layout=True)
            _plot_binary_step(ax_eng, t, result_df['ME available'],  'ME')
            _plot_binary_step(ax_eng, t, result_df['DG1 available'], 'DG1')
            _plot_binary_step(ax_eng, t, result_df['DG2 available'], 'DG2')
            _plot_binary_step(ax_eng, t, result_df['HSG available'], 'HSG')
            if 'available engine count' in result_df.columns:
                ax_eng.plot(t, result_df['available engine count'], linestyle='--', linewidth=1.0, label='Avail. count')
            ax_eng.set_title('Engine availability')
            ax_eng.set_ylim(-0.15, 4.25)
            ax_eng.set_yticks([0, 1, 2, 3, 4])
            ax_eng.set_ylabel('Availability')
            ax_eng.set_xlabel('Time (s)')
            ax_eng.grid(color='0.88', linestyle='-', linewidth=0.6)
            _maybe_fault_and_mode(ax_eng, t, result_df, include_fault_overlays, False)
            ax_eng.legend(loc='upper right', ncol=5, frameon=False, fontsize=8)

    else:
        # Full debugging layout retained for fault-heavy studies
        fig_1, axes = plt.subplots(nrows=3, ncols=3, figsize=(16, 10), constrained_layout=True)
        axes = axes.flatten()
        include_fault_overlays = has_faults
        include_mode_annotations = has_switches

        speed = np.sqrt(result_df['forward speed [m/s]']**2 + result_df['sideways speed [m/s]']**2)
        axes[0].plot(t, speed, label='Actual speed')
        axes[0].axhline(y=asset.ship_model.desired_speed, color='red', linestyle='--', linewidth=1.4, label='Desired speed')
        axes[0].set_title(f'{nm} speed tracking')
        _ax_style(axes[0], xlabel=True, ylabel='Speed (m/s)')
        _maybe_fault_and_mode(axes[0], t, result_df, include_fault_overlays, include_mode_annotations)
        axes[0].legend(loc='upper right', frameon=False)

        axes[1].plot(t, result_df['cross track error [m]'])
        axes[1].axhline(y=0.0, color='red', linestyle='--', linewidth=1.2)
        if hasattr(asset.ship_model, 'cross_track_error_tolerance'):
            tol = asset.ship_model.cross_track_error_tolerance
            axes[1].axhline(y=tol, color='0.45', linestyle=':', linewidth=1.0)
            axes[1].axhline(y=-tol, color='0.45', linestyle=':', linewidth=1.0)
        axes[1].set_title(f'{nm} cross-track error')
        _ax_style(axes[1], xlabel=True, ylabel='Error (m)')
        _maybe_fault_and_mode(axes[1], t, result_df, include_fault_overlays, include_mode_annotations)

        yaw = wrap_angle_deg(result_df['yaw angle [deg]'])
        axes[2].plot(t, yaw)
        axes[2].axhline(y=0.0, color='red', linestyle='--', linewidth=1.2)
        axes[2].set_title(f'{nm} yaw angle')
        _ax_style(axes[2], xlabel=True, ylabel='Yaw (deg)')
        axes[2].set_ylim(-180.0, 180.0)
        _maybe_fault_and_mode(axes[2], t, result_df, include_fault_overlays, include_mode_annotations)

        rud_lim = np.rad2deg(asset.ship_model.ship_machinery_model.rudder_ang_max)
        axes[3].plot(t, result_df['rudder angle [deg]'])
        axes[3].axhline(y=0.0, color='red', linestyle='--', linewidth=1.2)
        axes[3].axhline(y=rud_lim, color='0.45', linestyle=':', linewidth=1.0)
        axes[3].axhline(y=-rud_lim, color='0.45', linestyle=':', linewidth=1.0)
        axes[3].set_title(f'{nm} rudder activity')
        _ax_style(axes[3], xlabel=True, ylabel='Rudder (deg)')
        axes[3].set_ylim(-rud_lim, rud_lim)
        _maybe_fault_and_mode(axes[3], t, result_df, include_fault_overlays, include_mode_annotations)

        axes[4].plot(t, result_df['propeller shaft speed [rpm]'])
        axes[4].set_title(f'{nm} propeller shaft speed')
        _ax_style(axes[4], xlabel=True, ylabel='RPM')
        _maybe_fault_and_mode(axes[4], t, result_df, include_fault_overlays, include_mode_annotations)

        axes[5].plot(t, result_df['thrust force [kN]'])
        axes[5].set_title(f'{nm} thrust force')
        _ax_style(axes[5], xlabel=True, ylabel='Thrust (kN)')
        _maybe_fault_and_mode(axes[5], t, result_df, include_fault_overlays, include_mode_annotations)

        axp = axes[6]
        mech_power = _series_if_exists(result_df, 'power me [kw]')
        mech_avail = _series_if_exists(result_df, 'available power me [kw]')
        elec_power = _series_if_exists(result_df, 'power electrical [kw]')
        elec_avail = _series_if_exists(result_df, 'available power electrical [kw]')
        total_power = _series_if_exists(result_df, 'power [kw]')
        if mech_power is not None:
            axp.plot(t, mech_power, label='Mechanical power demand')
        if mech_avail is not None:
            axp.plot(t, mech_avail, '--', label='Mechanical power available')
        if elec_power is not None:
            axp.plot(t, elec_power, label='Electrical power demand')
        if elec_avail is not None:
            axp.plot(t, elec_avail, '--', label='Electrical power available')
        if total_power is not None:
            axp.plot(t, total_power, linewidth=1.5, alpha=0.8, label='Total power')
        axp.set_title(f'{nm} power demand and availability')
        _ax_style(axp, xlabel=True, ylabel='Power (kW)')
        _maybe_fault_and_mode(axp, t, result_df, include_fault_overlays, include_mode_annotations)
        axp.legend(loc='best', frameon=False, fontsize=9)

        axes[7].plot(t, result_df['fuel consumption [kg]'])
        axes[7].set_title(f'{nm} cumulative fuel consumption')
        _ax_style(axes[7], xlabel=True, ylabel='Fuel (kg)')
        _maybe_fault_and_mode(axes[7], t, result_df, include_fault_overlays, include_mode_annotations)

        axm = axes[8]
        if 'operating mode code' in result_df.columns:
            mode_code = result_df['operating mode code']
        elif 'operating mode' in result_df.columns:
            mode_code = _mode_to_code(result_df['operating mode'])
        else:
            mode_code = None
        if mode_code is not None:
            axm.step(t, mode_code, where='post', linewidth=1.5)
            axm.set_yticks([0, 1, 2, 3, 4, 5])
            axm.set_yticklabels(['PTO', 'MEC', 'PTI', 'PTI_1DG', 'MEC_BLK', 'DP_SAFE'])
            axm.set_title(f'{nm} machinery mode timeline')
            _ax_style(axm, xlabel=True, ylabel='Mode')
            _maybe_fault_and_mode(axm, t, result_df, include_fault_overlays, include_mode_annotations)
        else:
            axm.set_visible(False)

        if {'ME available', 'DG1 available', 'DG2 available', 'HSG available'}.issubset(result_df.columns):
            fig_eng, ax_eng = plt.subplots(figsize=(16, 4), constrained_layout=True)
            _plot_binary_step(ax_eng, t, result_df['ME available'],  'ME')
            _plot_binary_step(ax_eng, t, result_df['DG1 available'], 'DG1')
            _plot_binary_step(ax_eng, t, result_df['DG2 available'], 'DG2')
            _plot_binary_step(ax_eng, t, result_df['HSG available'], 'HSG')
            if 'available engine count' in result_df.columns:
                ax_eng.plot(t, result_df['available engine count'], linestyle='--', linewidth=1.0, label='Available engine count')
            ax_eng.set_title(f'{nm} engine availability and fault window')
            ax_eng.set_ylim(-0.15, 4.25)
            ax_eng.set_yticks([0, 1, 2, 3, 4])
            ax_eng.set_ylabel('Availability')
            ax_eng.set_xlabel('Time (s)')
            ax_eng.grid(color='0.85', linestyle='-', linewidth=0.5)
            _maybe_fault_and_mode(ax_eng, t, result_df, include_fault_overlays, include_mode_annotations)
            ax_eng.legend(loc='upper right', ncol=5, frameon=False)

    if plot_env_load:
        fig_w, aw = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True, constrained_layout=True)
        cols = ['wave force north [N]', 'wave force east [N]']
        max_value = result_df[cols].abs().to_numpy().max()
        aw[0].plot(t, result_df['wave force north [N]'])
        aw[0].set_title('Wave force north')
        _ax_style(aw[0], ylabel='Force (N)', ypct_series=result_df['wave force north [N]'])
        aw[0].set_ylim(-max_value, max_value)
        _maybe_fault_and_mode(aw[0], t, result_df, has_faults, False)
        aw[1].plot(t, result_df['wave force east [N]'])
        aw[1].set_title('Wave force east')
        _ax_style(aw[1], ypct_series=result_df['wave force east [N]'])
        aw[1].set_ylim(-max_value, max_value)
        _maybe_fault_and_mode(aw[1], t, result_df, has_faults, False)
        aw[2].plot(t, result_df['wave moment [Nm]'])
        aw[2].set_title('Wave moment')
        _ax_style(aw[2], xlabel=True, ylabel='Moment (Nm)', ypct_series=result_df['wave moment [Nm]'])
        _maybe_fault_and_mode(aw[2], t, result_df, has_faults, False)

    if show:
        plt.show()


def plot_ship_and_real_map(assets, result_dfs, map_gdfs=None, show=False, no_title=False, profile="paper",
                           crop_to_data=True, margin_m=4000.0, show_ship_drawings=False,
                           show_mode_switches=False, show_grid=True, show_scalebar=True,
                           asset_display="ais", ais_history_points=18,
                           ais_vector_m=None, ais_marker_m=None,
                           show_full_trajectory=None, annotate_assets=True):
    """
    Plot simulated assets on the real map.

    asset_display:
        "ais"  -> AIS-style current vessel body, short dotted history behind it,
                  and a short heading vector in front of it.
        "line" -> original full-line trajectory view.

    For an AIS-like plot, the full trajectory is hidden by default so the figure
    does not become a collection of long coloured lines. Set show_full_trajectory=True
    if you still want a faint full-track reference.
    """
    if profile == "paper":
        fig, ax = plt.subplots(figsize=(7.8, 5.1))
        title_fs, label_fs, tick_fs, legend_fs = 12, 10, 9, 9
        traj_lw, route_lw = 1.8, 1.0
    else:
        fig, ax = plt.subplots(figsize=(12, 8))
        title_fs, label_fs, tick_fs, legend_fs = 18, 14, 12, 12
        traj_lw, route_lw = 2.2, 1.2

    if show_full_trajectory is None:
        show_full_trajectory = (asset_display != "ais")

    if map_gdfs is not None:
        frame_gdf, ocean_gdf, land_gdf, coast_gdf, water_gdf, ferry_route_gdf = map_gdfs
        # More chart-like visual hierarchy: water first, land on top, thin coastlines.
        if ocean_gdf is not None and not ocean_gdf.empty:
            ocean_gdf.plot(ax=ax, facecolor="#a9c9ea", edgecolor="none", zorder=0) #b8dce8
        if land_gdf is not None and not land_gdf.empty:
            land_gdf.plot(ax=ax, facecolor="#f7f7f5", edgecolor="#c8c8c8", linewidth=0.30, zorder=1)
        if water_gdf is not None and not water_gdf.empty:
            water_gdf.plot(ax=ax, facecolor="#a9c9ea", edgecolor="#9cc8d3", linewidth=0.25, alpha=0.98, zorder=2)
        if coast_gdf is not None and not coast_gdf.empty:
            coast_gdf.plot(ax=ax, color="#8fb8c1", linewidth=0.55, zorder=3)
        if ferry_route_gdf is not None and not ferry_route_gdf.empty:
            ferry_route_gdf.plot(ax=ax, color="#f39c12", linewidth=1.0, linestyle=(0, (5, 3)), alpha=0.85, zorder=4)

    palette = ['#2c7fb8', '#e34a33', '#31a354', '#bdbf00', '#d65db1', '#8c6d31', '#636363', '#17becf', '#756bb1']
    all_e = []
    all_n = []
    asset_records = []

    for i, asset in enumerate(assets):
        c = palette[i % len(palette)]
        df = result_dfs[i]
        e = df['east position [m]'].to_numpy(dtype=float)
        n = df['north position [m]'].to_numpy(dtype=float)
        all_e.append(e)
        all_n.append(n)

        if show_full_trajectory:
            alpha = 0.28 if asset_display == "ais" else 0.98
            ax.plot(e, n, lw=traj_lw, alpha=alpha, color=c, zorder=5, label=asset.info.name_tag)

        if asset.ship_model.auto_pilot is not None:
            re = np.asarray(asset.ship_model.auto_pilot.navigate.east, dtype=float)
            rn = np.asarray(asset.ship_model.auto_pilot.navigate.north, dtype=float)
            all_e.append(re)
            all_n.append(rn)
            route_color = '#f39c12' if asset_display == "ais" else c
            ax.plot(re, rn, linestyle=(0, (5, 3)), lw=route_lw, alpha=0.65,
                    color=route_color, zorder=4, label='_nolegend_')
            if profile != "paper" and asset_display != "ais":
                ax.scatter(re, rn, marker='x', s=20, linewidths=1.0, color=c, zorder=6)

        if show_ship_drawings and asset_display != "ais":
            for x, y in zip(asset.ship_model.ship_drawings[1], asset.ship_model.ship_drawings[0]):
                ax.plot(x, y, color=c, lw=0.8, zorder=7)

        yaw_deg = df['yaw angle [deg]'].to_numpy(dtype=float) if 'yaw angle [deg]' in df.columns else None
        asset_records.append((asset, e, n, yaw_deg, c))

    if crop_to_data and all_e and all_n:
        ecat = np.concatenate(all_e)
        ncat = np.concatenate(all_n)
        ax.set_xlim(np.nanmin(ecat) - margin_m, np.nanmax(ecat) + margin_m)
        ax.set_ylim(np.nanmin(ncat) - margin_m, np.nanmax(ncat) + margin_m)
    elif map_gdfs is not None:
        minx, miny, maxx, maxy = map_gdfs[0].total_bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

    # AIS symbols are drawn after limits are fixed so their size adapts to the displayed map extent.
    if asset_display == "ais":
        for asset, e, n, yaw_deg, c in asset_records:
            idx = _finite_last_index(e, n)
            if idx is None:
                continue
            yaw_rad = _infer_yaw_rad(e, n, idx, yaw_deg)
            _draw_ais_asset(
                ax, e, n, yaw_rad, c, label=asset.info.name_tag, asset=asset,
                history_points=ais_history_points, vector_m=ais_vector_m,
                marker_length_m=ais_marker_m, zorder=9,
                annotate=(annotate_assets and profile != "paper")
            )

    ax.set_aspect('equal', adjustable='box')
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax.tick_params(axis='both', which='major', labelsize=tick_fs, length=3)
    ax.xaxis.get_offset_text().set_fontsize(tick_fs - 1)
    ax.yaxis.get_offset_text().set_fontsize(tick_fs - 1)

    ax.set_xlabel('East position (m)', fontsize=label_fs)
    ax.set_ylabel('North position (m)', fontsize=label_fs)
    if not no_title:
        ax.set_title('Ship trajectory', fontsize=title_fs)

    if show_grid:
        ax.grid(True, linewidth=0.45, alpha=0.30, color='0.65')

    if show_scalebar:
        _add_scalebar(ax)

    # Legend: keep it compact. AIS mode uses the vessel colours, not the full track lines.
    if asset_display == "ais":
        handles = [Line2D([0], [0], color=palette[i % len(palette)], lw=1.6,
                          linestyle=(0, (3, 3)), label=getattr(assets[i].info, 'name_tag', f'Ship {i+1}'))
                   for i in range(len(assets))]
        if any(getattr(a.ship_model, 'auto_pilot', None) is not None for a in assets):
            handles.append(Line2D([0], [0], color='#f39c12', lw=route_lw,
                                  linestyle=(0, (5, 3)), label='Planned route'))
        handles.append(Line2D([0], [0], color='0.12', lw=1.1, label='Heading vector'))
        lg = ax.legend(handles=handles, loc='upper right', frameon=True, fontsize=legend_fs)
    elif len(assets) == 1 and assets[0].ship_model.auto_pilot is not None:
        actual = Line2D([0], [0], color='#1f77b4', lw=traj_lw, label='Actual trajectory')
        planned = Line2D([0], [0], color='#1f77b4', lw=route_lw, linestyle='--', label='Planned route')
        lg = ax.legend(handles=[actual, planned], loc='upper right', frameon=True, fontsize=legend_fs)
    else:
        lg = ax.legend(loc='upper right', frameon=True, fontsize=legend_fs)
    lg.get_frame().set_alpha(0.93)

    if show_mode_switches and len(result_dfs) > 0:
        try:
            own_df = result_dfs[0]
            if 'operating mode' in own_df.columns and own_df['operating mode'].astype(str).nunique() > 1:
                modes = own_df['operating mode'].astype(str).to_numpy()
                switch_idx = np.where(modes[1:] != modes[:-1])[0] + 1
                e = own_df['east position [m]'].to_numpy(dtype=float)
                n = own_df['north position [m]'].to_numpy(dtype=float)
                ax.scatter(e[switch_idx], n[switch_idx], marker='o', s=20, zorder=15, color='k')
                for idx in switch_idx:
                    ax.annotate(modes[idx], (e[idx], n[idx]), textcoords='offset points', xytext=(4, 4), fontsize=7)
        except Exception:
            pass

    fig.tight_layout(pad=0.2)
    if show:
        plt.show()
    return fig, ax

