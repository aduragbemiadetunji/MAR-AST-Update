# mc_runner.py
import numpy as np
from typing import Dict, Any
from mc_scenarios import Scenario, engine_status_at_time
from test_beds.ship_simu_test.mode_changer_helper import resilience_supervisor, apply_safe_state_overrides

def run_episode(
    env,
    assets,
    scenario: Scenario,
    machinery_config,
    main_engine_capacity: float,
    diesel_gen_capacity: float,
    min_dwell_s: float = 60.0,
) -> Dict[str, Any]:
    env.reset()

    ship = assets[0].ship_model
    mach = ship.ship_machinery_model

    # Keep a "normal" desired speed to restore after DP_SAFE
    normal_desired_speed = float(getattr(ship, "desired_speed", 4.0))

    running_time = 0.0
    last_mode_change_time = 0.0

    # for summary
    dp_safe_time = 0.0
    mode_switches = 0

    while running_time < ship.int.sim_time and (env.stop is False):

        # 1) inject faults (health outside window)
        engine_status = engine_status_at_time(scenario, running_time)

        # 2) define demand (DON'T use achieved propulsion power as "required")
        required_hotel_power = machinery_config.hotel_load

        # if still simulating, we assume "ship wants to keep moving"
        # (this avoids DP_SAFE becoming "feasible forever" after you set desired_speed=0 inside DP_SAFE)
        required_propulsion_power = 0.6 * main_engine_capacity

        # 3) dwell time
        time_in_mode = running_time - last_mode_change_time

        # 4) supervisor decision
        decision = resilience_supervisor(
            current_mode=mach.operating_mode,
            engine_status=engine_status,
            required_propulsion_power=required_propulsion_power,
            required_hotel_power=required_hotel_power,
            me_capacity=main_engine_capacity,
            dg_capacity=diesel_gen_capacity,
            min_dwell_s=min_dwell_s,
            time_in_mode_s=time_in_mode,
        )

        if decision.action in ("SWITCH_MODE", "ENTER_SAFE_STATE"):
            if mach.operating_mode != decision.recommended_mode:
                mach.operating_mode = decision.recommended_mode
                last_mode_change_time = running_time
                mode_switches += 1

        # 5) enforce DP_SAFE drift OR restore normal desired speed
        mode_now = str(mach.operating_mode).upper()
        if mode_now == "DP_SAFE":
            dp_safe_time += ship.int.dt if hasattr(ship.int, "dt") else 0.0
            apply_safe_state_overrides(ship)   # sets desired_speed=0, omega=0, etc.
        else:
            # restore propulsion intent after leaving DP_SAFE
            if hasattr(ship, "desired_speed"):
                ship.desired_speed = normal_desired_speed

        # 6) step sim AFTER decision + overrides
        env.step()

        # 7) update time
        running_time = max(a.ship_model.int.time for a in assets)

    # episode summary
    return {
        "seed": scenario.seed,
        "fault_start_s": scenario.fault_start_s,
        "fault_duration_s": scenario.fault_duration_s,
        "fault_mask": str(scenario.engine_fault_mask),
        "mode_switches": mode_switches,
        "dp_safe_time_s": dp_safe_time,
        "stopped_early": bool(env.stop),
        "t_end_s": float(running_time),
        "final_mode": str(mach.operating_mode),
    }
