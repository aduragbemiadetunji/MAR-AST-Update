import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42    # Use TrueType instead of Type 3
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial' #'DejaVu Sans'  # Or: 'Arial', 'Helvetica'
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.lines import Line2D

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



# def plot_ship_status(asset, result_df, plot_env_load=True, show=False):
#     def _ax_style(ax, *, xlabel=False, ylabel=None, xlim_left=0, ypct_series=None):
#         ax.grid(color='0.85', linestyle='-', linewidth=0.5)
#         ax.set_xlim(left=xlim_left)
#         if ylabel:
#             ax.set_ylabel(ylabel)
#         if xlabel:
#             ax.set_xlabel('Time (s)')
#         if ypct_series is not None:
#             # percentile-based y-limits to avoid one spike flattening the trace
#             lo, hi = np.nanpercentile(ypct_series, [1, 99])
#             if np.isfinite(lo) and np.isfinite(hi) and lo != hi:
#                 pad = 0.05 * (hi - lo)
#                 ax.set_ylim(lo - pad, hi + pad)

#     # Global readability
#     plt.rcParams.update({
#         "font.size": 11,
#         "axes.titlesize": 12,
#         "axes.labelsize": 11,
#         "xtick.labelsize": 10,
#         "ytick.labelsize": 10,
#         "lines.linewidth": 1.1,
#     })

#     # ---------- FIGURE 1: SHIP STATUS ----------
#     fig_1, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 8), constrained_layout=True)
#     axes = axes.flatten()
#     plt.figure(fig_1.number)

#     # Center plotting
#     try:
#         center_plot_window()
#     except Exception:
#         pass

#     t = result_df['time [s]']
#     nm = str(asset.info.name_tag)

#     # 1. Speed (use resultant to match your code)
#     speed = np.sqrt(result_df['forward speed [m/s]']**2 + result_df['sideways speed [m/s]']**2)
#     axes[0].plot(t, speed, label='Speed')
#     axes[0].axhline(y=asset.ship_model.desired_speed, color='red', linestyle='--', linewidth=1.5, label='Desired Forward Speed')
#     axes[0].set_title(f'{nm} Speed')
#     _ax_style(axes[0], xlabel=True, ylabel='Speed (m/s)')
#     axes[0].legend(loc='upper right', frameon=False)

#     # 2. Yaw angle
#     axes[1].plot(t, wrap_angle_deg(result_df['yaw angle [deg]']))
#     axes[1].axhline(y=0.0, color='red', linestyle='--', linewidth=1.5)
#     axes[1].set_title(f'{nm} Yaw Angle')
#     _ax_style(axes[1], xlabel=True, ylabel='Yaw angle (deg)')
#     axes[1].set_ylim(-180.0, 180.0)

#     # 3. Rudder angle
#     axes[2].plot(t, result_df['rudder angle [deg]'])
#     axes[2].axhline(y=0.0, color='red', linestyle='--', linewidth=1.5)
#     axes[2].set_title(f'{nm} Rudder Angle')
#     _ax_style(axes[2], xlabel=True, ylabel='Rudder angle (deg)')
#     axes[2].set_ylim(-np.rad2deg(asset.ship_model.ship_machinery_model.rudder_ang_max), np.rad2deg(asset.ship_model.ship_machinery_model.rudder_ang_max))

#     # 4. Cross-track error
#     axes[3].plot(t, result_df['cross track error [m]'])
#     axes[3].axhline(y=0.0, color='red', linestyle='--', linewidth=1.5)
#     axes[3].set_title(f'{nm} Cross-Track Error')
#     _ax_style(axes[3], xlabel=True, ylabel='Cross-track error (m)')
#     axes[3].set_ylim(-asset.ship_model.cross_track_error_tolerance, asset.ship_model.cross_track_error_tolerance)

#     # 5. Propeller shaft speed
#     axes[4].plot(t, result_df['propeller shaft speed [rpm]'])
#     axes[4].set_title(f'{nm} Propeller Shaft Speed')
#     _ax_style(axes[4], xlabel=True, ylabel='Shaft speed (rpm)')
    
#     # 6. Thrust Force
#     axes[5].plot(t, result_df['thrust force [kN]'])
#     axes[5].set_title(f'{nm} Thrust force')
#     _ax_style(axes[5], xlabel=True, ylabel='Thrust force (kN)')

#     # 7. Power vs available power (mode-dependent)
#     ax6 = axes[6]
#     mode = asset.ship_model.ship_machinery_model.operating_mode
#     if mode in ('PTO', 'MEC'):
#         ax6.plot(t, result_df['power me [kw]'], label='Power')
#         ax6.plot(t, result_df['available power me [kw]'], color='red', linestyle='--', label='Available Power')
#         ax6.set_title(f'{nm} Power vs Available Mechanical Power')
#         _ax_style(ax6, xlabel=True, ylabel='Power (kW)')
#         ax6.legend(frameon=False)
#     elif mode == 'PTI':
#         ax6.plot(t, result_df['power electrical [kw]'], label='Power')
#         ax6.plot(t, result_df['available power electrical [kw]'], color='red', linestyle='--', label='Available Power')
#         ax6.set_title(f'{nm} Power vs Available Electrical Power')
#         _ax_style(ax6, xlabel=True, ylabel='Power (kW)')
#         ax6.legend(frameon=False)
#     else:
#         ax6.set_visible(False)

#     # 8. Fuel consumption
#     axes[7].plot(t, result_df['fuel consumption [kg]'])
#     axes[7].set_title(f'{nm} Fuel Consumption')
#     _ax_style(axes[7], xlabel=True, ylabel='Fuel (kg)')

#     # ---------- FIGURES 2–4: ENVIRONMENT LOADS (optional) ----------
#     if plot_env_load:
#         # --- WAVES: Fx, Fy, Mz ---
#         fig_w, aw = plt.subplots(1, 3, figsize=(16, 5), sharex=True, constrained_layout=True)
#         cols = ['wave force north [N]', 'wave force east [N]']
#         max_value = result_df[cols].abs().to_numpy().max()
        
#         aw[0].plot(t, result_df['wave force north [N]'])
#         aw[0].set_title('Wave Force North'); _ax_style(aw[0], ylabel='Force (N)', ypct_series=result_df['wave force north [N]'])
#         aw[0].set_ylim(-max_value, max_value)

#         aw[1].plot(t, result_df['wave force east [N]'])
#         aw[1].set_title('Wave Force East'); _ax_style(aw[1], ypct_series=result_df['wave force east [N]'])
#         aw[1].set_ylim(-max_value, max_value)

#         aw[2].plot(t, result_df['wave moment [Nm]'])
#         aw[2].set_title('Wave Moment'); _ax_style(aw[2], xlabel=True, ylabel='Moment (Nm)', ypct_series=result_df['wave moment [Nm]'])

#         fig_w.suptitle(f'Wave loads on {nm}', y=1.02)

#         # --- WIND: speed, dir, Fx, Fy, Mz ---
#         fig_wind, axw = plt.subplots(2, 3, figsize=(18, 8), sharex=True, constrained_layout=True)
#         axw = axw.ravel()
        
#         cols = ['wind force north [N]', 'wind force east [N]']
#         max_value = result_df[cols].abs().to_numpy().max()

#         axw[0].plot(t, result_df['wind speed [m/s]'])
#         axw[0].set_title('Wind Speed'); _ax_style(axw[0], ylabel='Speed (m/s)', ypct_series=result_df['wind speed [m/s]'])

#         axw[1].plot(t, result_df['wind dir [deg]'])
#         axw[1].set_title('Wind Direction'); _ax_style(axw[1], ypct_series=result_df['wind dir [deg]'])
#         axw[1].set_ylim(-180, 180)

#         axw[2].axis('off')  # spacer to make a clean 2x3 grid

#         axw[3].plot(t, result_df['wind force north [N]'])
#         axw[3].set_title('Wind Force North'); _ax_style(axw[3], ylabel='Force (N)', ypct_series=result_df['wind force north [N]'])
#         axw[3].set_ylim(-max_value, max_value)

#         axw[4].plot(t, result_df['wind force east [N]'])
#         axw[4].set_title('Wind Force East'); _ax_style(axw[4], ypct_series=result_df['wind force east [N]'])
#         axw[4].set_ylim(-max_value, max_value)

#         axw[5].plot(t, result_df['wind moment [Nm]'])
#         axw[5].set_title('Wind Moment'); _ax_style(axw[5], xlabel=True, ylabel='Moment (Nm)', ypct_series=result_df['wind moment [Nm]'])

#         fig_wind.suptitle(f'Wind field & loads on {nm}', y=1.02)

#         # --- CURRENT: speed, dir ---
#         fig_c, ac = plt.subplots(1, 2, figsize=(12, 5), sharex=True, constrained_layout=True)

#         ac[0].plot(t, result_df['current speed [m/s]'])
#         ac[0].set_title('Current Speed'); _ax_style(ac[0], ylabel='Speed (m/s)', ypct_series=result_df['current speed [m/s]'])

#         ac[1].plot(t, result_df['current dir [deg]'])
#         ac[1].set_title('Current Direction'); _ax_style(ac[1], xlabel=True, ylabel='Angle in NED (deg)')
#         ac[1].set_ylim(-180, 180)

#         fig_c.suptitle(f'Current field on {nm}', y=1.02)

#     if show:
#         plt.show()



def plot_ship_status(asset, result_df, plot_env_load=True, show=False):
    def _ax_style(ax, *, xlabel=False, ylabel=None, xlim_left=0, ypct_series=None):
        ax.grid(color='0.85', linestyle='-', linewidth=0.5)
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
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.2,
    })

    t = result_df['time [s]'].to_numpy()
    nm = str(asset.info.name_tag)

    # -------------------- Figure 1: primary ship response --------------------
    fig_1, axes = plt.subplots(nrows=3, ncols=3, figsize=(16, 10), constrained_layout=True)
    axes = axes.flatten()

    # 1. Speed
    speed = np.sqrt(result_df['forward speed [m/s]']**2 + result_df['sideways speed [m/s]']**2)
    axes[0].plot(t, speed, label='Actual speed')
    axes[0].axhline(
        y=asset.ship_model.desired_speed,
        color='red', linestyle='--', linewidth=1.4,
        label='Desired speed'
    )
    axes[0].set_title(f'{nm} speed tracking')
    _ax_style(axes[0], xlabel=True, ylabel='Speed (m/s)')
    _shade_fault_windows(axes[0], t, result_df)
    _annotate_mode_switches(axes[0], t, result_df)
    axes[0].legend(loc='upper right', frameon=False)

    # 2. Cross-track error
    axes[1].plot(t, result_df['cross track error [m]'], label='Cross-track error')
    axes[1].axhline(y=0.0, color='red', linestyle='--', linewidth=1.2)
    if hasattr(asset.ship_model, 'cross_track_error_tolerance'):
        tol = asset.ship_model.cross_track_error_tolerance
        axes[1].axhline(y=tol, color='0.45', linestyle=':', linewidth=1.0)
        axes[1].axhline(y=-tol, color='0.45', linestyle=':', linewidth=1.0)
    axes[1].set_title(f'{nm} cross-track error')
    _ax_style(axes[1], xlabel=True, ylabel='Error (m)')
    _shade_fault_windows(axes[1], t, result_df)
    _annotate_mode_switches(axes[1], t, result_df)

    # 3. Yaw angle
    yaw = wrap_angle_deg(result_df['yaw angle [deg]'])
    axes[2].plot(t, yaw)
    axes[2].axhline(y=0.0, color='red', linestyle='--', linewidth=1.2)
    axes[2].set_title(f'{nm} yaw angle')
    _ax_style(axes[2], xlabel=True, ylabel='Yaw (deg)')
    axes[2].set_ylim(-180.0, 180.0)
    _shade_fault_windows(axes[2], t, result_df)
    _annotate_mode_switches(axes[2], t, result_df)

    # 4. Rudder angle
    rud_lim = np.rad2deg(asset.ship_model.ship_machinery_model.rudder_ang_max)
    axes[3].plot(t, result_df['rudder angle [deg]'])
    axes[3].axhline(y=0.0, color='red', linestyle='--', linewidth=1.2)
    axes[3].axhline(y=rud_lim, color='0.45', linestyle=':', linewidth=1.0)
    axes[3].axhline(y=-rud_lim, color='0.45', linestyle=':', linewidth=1.0)
    axes[3].set_title(f'{nm} rudder activity')
    _ax_style(axes[3], xlabel=True, ylabel='Rudder (deg)')
    axes[3].set_ylim(-rud_lim, rud_lim)
    _shade_fault_windows(axes[3], t, result_df)
    _annotate_mode_switches(axes[3], t, result_df)

    # 5. Shaft speed
    axes[4].plot(t, result_df['propeller shaft speed [rpm]'])
    axes[4].set_title(f'{nm} propeller shaft speed')
    _ax_style(axes[4], xlabel=True, ylabel='RPM')
    _shade_fault_windows(axes[4], t, result_df)
    _annotate_mode_switches(axes[4], t, result_df)

    # 6. Thrust
    axes[5].plot(t, result_df['thrust force [kN]'])
    axes[5].set_title(f'{nm} thrust force')
    _ax_style(axes[5], xlabel=True, ylabel='Thrust (kN)')
    _shade_fault_windows(axes[5], t, result_df)
    _annotate_mode_switches(axes[5], t, result_df)

    # 7. Power demand vs available power
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
    _shade_fault_windows(axp, t, result_df)
    _annotate_mode_switches(axp, t, result_df)
    axp.legend(loc='best', frameon=False, fontsize=9)

    # 8. Fuel
    axes[7].plot(t, result_df['fuel consumption [kg]'])
    axes[7].set_title(f'{nm} cumulative fuel consumption')
    _ax_style(axes[7], xlabel=True, ylabel='Fuel (kg)')
    _shade_fault_windows(axes[7], t, result_df)
    _annotate_mode_switches(axes[7], t, result_df)

    # 9. Operating mode as step signal
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
        _shade_fault_windows(axm, t, result_df)
        _annotate_mode_switches(axm, t, result_df)
    else:
        axm.set_visible(False)

    # -------------------- Figure 2: engine availability --------------------
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
        _shade_fault_windows(ax_eng, t, result_df)
        _annotate_mode_switches(ax_eng, t, result_df)
        ax_eng.legend(loc='upper right', ncol=5, frameon=False)

    # -------------------- Figure 3: environment loads (optional) --------------------
    if plot_env_load:
        fig_w, aw = plt.subplots(1, 3, figsize=(16, 5), sharex=True, constrained_layout=True)

        cols = ['wave force north [N]', 'wave force east [N]']
        max_value = result_df[cols].abs().to_numpy().max()

        aw[0].plot(t, result_df['wave force north [N]'])
        aw[0].set_title('Wave force north')
        _ax_style(aw[0], ylabel='Force (N)', ypct_series=result_df['wave force north [N]'])
        aw[0].set_ylim(-max_value, max_value)
        _shade_fault_windows(aw[0], t, result_df)

        aw[1].plot(t, result_df['wave force east [N]'])
        aw[1].set_title('Wave force east')
        _ax_style(aw[1], ypct_series=result_df['wave force east [N]'])
        aw[1].set_ylim(-max_value, max_value)
        _shade_fault_windows(aw[1], t, result_df)

        aw[2].plot(t, result_df['wave moment [Nm]'])
        aw[2].set_title('Wave moment')
        _ax_style(aw[2], xlabel=True, ylabel='Moment (Nm)', ypct_series=result_df['wave moment [Nm]'])
        _shade_fault_windows(aw[2], t, result_df)

        fig_w.suptitle(f'Wave loads on {nm}', y=1.02)

    if show:
        plt.show()


def plot_ship_and_real_map(assets, result_dfs, map_gdfs=None, show=False, no_title=False):
    fig, ax = plt.subplots(figsize=(19, 12))  # temp; we’ll resize to aspect
    
    if map_gdfs is not None:
        frame_gdf, ocean_gdf, land_gdf, coast_gdf, water_gdf, ferry_route_gdf = map_gdfs
        if not land_gdf.empty:
            land_gdf.plot(ax=ax, facecolor="#e6e6e6", edgecolor="#bbbbbb", linewidth=0.3, zorder=0)
        if not ocean_gdf.empty:
            ocean_gdf.plot(ax=ax, facecolor="#a0c8f0", edgecolor="none", zorder=1)
        if not water_gdf.empty:
            water_gdf.plot(ax=ax, facecolor="#a0c8f0", edgecolor="#74a8d8", linewidth=0.4, alpha=0.95, zorder=2)
        if not coast_gdf.empty:
            coast_gdf.plot(ax=ax, color="#a0c8f0", linewidth=1.2, zorder=3)
        if not ferry_route_gdf.empty:
            ferry_route_gdf.plot(ax=ax, color="#432466", linewidth=1.2, linestyle='--', zorder=3)

        # --- fit to frame & keep aspect ---
        minx, miny, maxx, maxy = frame_gdf.total_bounds
        dx, dy = (maxx - minx), (maxy - miny)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

        # Resize figure to match map aspect (no letterbox)
        fig_w = 19.0
        fig_h = fig_w * (dy / dx) if dx > 0 else 12.0
        fig.set_size_inches(fig_w, fig_h, forward=True)

    # Full-bleed canvas
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_position([0, 0, 1, 1])
    
    # -------- enforce 1:1 scale from DATA --------
    # If no map was set, autoscale to the plotted data first
    ax.relim(); ax.autoscale_view()
    # Now lock aspect so 1 unit in x == 1 unit in y
    ax.set_aspect('equal', adjustable='box')   # robust when limits come from data

    # --- ships & routes ---
    palette = ['#1f77b4', '#d62728', '#2ca02c', '#bcbd22', '#e377c2', '#8c564b', '#000000', '#7f7f7f', '#17becf']
    for i, asset in enumerate(assets):
        c = palette[i % len(palette)]
        ax.plot(result_dfs[i]['east position [m]'].to_numpy(),
                result_dfs[i]['north position [m]'].to_numpy(),
                lw=2, alpha=0.95, label=asset.info.name_tag, zorder=5)
        if asset.ship_model.auto_pilot is not None:
            ax.plot(asset.ship_model.auto_pilot.navigate.east,
                    asset.ship_model.auto_pilot.navigate.north,
                    linestyle='--', lw=1.2, alpha=0.85, color=c, zorder=5)
            ax.scatter(asset.ship_model.auto_pilot.navigate.east,
                    asset.ship_model.auto_pilot.navigate.north,
                    marker='x', s=28, linewidths=1.2, color=c, zorder=6)
        for x, y in zip(asset.ship_model.ship_drawings[1], asset.ship_model.ship_drawings[0]):
            ax.plot(x, y, color=c, lw=1.0, zorder=7)

    lg = ax.legend(loc='upper right', frameon=True, fontsize=24)
    lg.get_frame().set_alpha(0.95)
    
    ax.set_axis_on()
    fig.subplots_adjust(left=0.06, right=0.99, top=0.96, bottom=0.07)
    if not no_title:
        ax.set_title('Ship Trajectory') 
        ax.title.set_size(24)                                     # title font size
    ax.set_xlabel('East position (m)')
    ax.set_ylabel('North position (m)')
    # ax.grid(True, linewidth=0.5, alpha=0.4)
    
        # >>> add these lines <<<
    ax.tick_params(axis='both', which='major', labelsize=24)  # tick label size
    ax.xaxis.label.set_size(24)                               # x-label font size
    ax.yaxis.label.set_size(24)                               # y-label font size


        # Optional: mark mode switch points for the own ship
    try:
        own_df = result_dfs[0]
        if 'operating mode' in own_df.columns:
            modes = own_df['operating mode'].astype(str).to_numpy()
            switch_idx = np.where(modes[1:] != modes[:-1])[0] + 1

            e = own_df['east position [m]'].to_numpy()
            n = own_df['north position [m]'].to_numpy()

            ax.scatter(
                e[switch_idx], n[switch_idx],
                marker='o', s=28, zorder=6, label='Mode switch'
            )

            for idx in switch_idx:
                ax.annotate(
                    modes[idx],
                    (e[idx], n[idx]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8
                )
    except Exception:
        pass

    if show:
        plt.show()

# def plot_ship_and_real_map(
#         assets,
#         result_dfs,
#         map_gdfs=None,
#         show=False,
#         no_title=False,
#         fig_width=2.5,      # in inches – good for single-column
#         dpi=500,            # higher DPI for LaTeX
#     ):
#     # --- base figure ---------------------------------------------------------
#     fig, ax = plt.subplots(figsize=(fig_width, fig_width))  # temp square; we’ll fix aspect
#     fig.set_dpi(dpi)

#     # --- map layers ----------------------------------------------------------
#     if map_gdfs is not None:
#         frame_gdf, ocean_gdf, land_gdf, coast_gdf, water_gdf, ferry_route_gdf = map_gdfs

#         if not land_gdf.empty:
#             land_gdf.plot(ax=ax, facecolor="#e6e6e6",
#                           edgecolor="#b0b0b0", linewidth=0.4, zorder=0)
#         if not ocean_gdf.empty:
#             ocean_gdf.plot(ax=ax, facecolor="#a0c8f0",
#                            edgecolor="none", zorder=1)
#         if not water_gdf.empty:
#             water_gdf.plot(ax=ax, facecolor="#a0c8f0",
#                            edgecolor="#5c8fc4", linewidth=0.5,
#                            alpha=0.98, zorder=2)
#         if not coast_gdf.empty:
#             coast_gdf.plot(ax=ax, color="#2f7f3f",
#                            linewidth=1.4, zorder=3)
#         if not ferry_route_gdf.empty:
#             ferry_route_gdf.plot(ax=ax, color="#432466", linewidth=1.2, linestyle='--', zorder=3)

#         # fit to frame & keep aspect
#         minx, miny, maxx, maxy = frame_gdf.total_bounds
#         dx, dy = (maxx - minx), (maxy - miny)
#         ax.set_xlim(minx, maxx)
#         ax.set_ylim(miny, maxy)

#         # resize figure to preserve map aspect at the chosen width
#         if dx > 0:
#             fig_h = fig_width * (dy / dx)
#             fig.set_size_inches(fig_width, fig_h, forward=True)

#     # full-bleed canvas during drawing
#     ax.set_axis_off()
#     fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
#     ax.set_position([0, 0, 1, 1])

#     # autoscale to data, then enforce 1:1 scale
#     ax.relim()
#     ax.autoscale_view()
#     ax.set_aspect("equal", adjustable="box")

#     # --- ship trajectories ---------------------------------------------------
#     own_color   = "#0c3c78"   # dark blue for actual ship path
#     route_color = "#d90808"   # orange for mission trajectory / waypoints

#     own_handle = None
#     route_handle = None

#     for i, asset in enumerate(assets):
#         # actual sailed track
#         line, = ax.plot(
#             result_dfs[i]["east position [m]"].to_numpy(),
#             result_dfs[i]["north position [m]"].to_numpy(),
#             color=own_color,
#             lw=2.4,
#             alpha=0.98,
#             zorder=5,
#             label="Own ship" if own_handle is None else "_nolegend_",
#         )
#         if own_handle is None:
#             own_handle = line

#         # planned/mission trajectory from autopilot
#         if asset.ship_model.auto_pilot is not None:
#             route_line, = ax.plot(
#                 asset.ship_model.auto_pilot.navigate.east,
#                 asset.ship_model.auto_pilot.navigate.north,
#                 linestyle="--",
#                 lw=1.8,
#                 alpha=0.95,
#                 color=route_color,
#                 zorder=5,
#                 label="Mission trajectory" if route_handle is None else "_nolegend_",
#             )
#             ax.scatter(
#                 asset.ship_model.auto_pilot.navigate.east,
#                 asset.ship_model.auto_pilot.navigate.north,
#                 marker="x",
#                 s=30,
#                 linewidths=1.6,
#                 color=route_color,
#                 zorder=6,
#             )
#             if route_handle is None:
#                 route_handle = route_line

#         # ship outlines along the path
#         for x, y in zip(asset.ship_model.ship_drawings[1],
#                         asset.ship_model.ship_drawings[0]):
#             ax.plot(x, y, color=own_color, lw=1.0, zorder=7)

#     # --- axes, labels, legend -----------------------------------------------
#     ax.set_axis_on()
#     # fig.subplots_adjust(left=0.075, right=0.45, top=0.45, bottom=0.075)
#     fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.12)

#     title_fs = 12
#     label_fs = 11
#     tick_fs  = 10
#     legend_fs = 9

#     if not no_title:
#         ax.set_title("Ship trajectory", fontsize=title_fs)

#     ax.set_xlabel("East position (m)", fontsize=label_fs)
#     ax.set_ylabel("North position (m)", fontsize=label_fs)
#     # --- scientific notation for coordinates ---
#     ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
#     ax.xaxis.get_offset_text().set_fontsize(9)
#     ax.yaxis.get_offset_text().set_fontsize(9)

#     ax.tick_params(axis="both", which="major", labelsize=tick_fs, length=3)


#     # legend (only if we have handles)
#     handles = [h for h in (own_handle, route_handle) if h is not None]
#     labels = [h.get_label() for h in handles]
#     if handles:
#         lg = ax.legend(handles, labels,
#                        loc="upper right",
#                        frameon=True,
#                        fontsize=legend_fs)
#         lg.get_frame().set_alpha(0.95)

#     if show:
#         plt.show()
        
#     # === resize figure for paper ===
#     target_width = 5.0  # inches, change as you like (e.g. 2.5, 3.5, ...)
#     # preserve the current data aspect ratio
#     x0, x1 = ax.get_xlim()
#     y0, y1 = ax.get_ylim()
#     aspect = (y1 - y0) / (x1 - x0)

#     fig.set_size_inches(target_width, target_width * aspect, forward=True)
#     fig.tight_layout(pad=0.05)

#     return fig, ax
