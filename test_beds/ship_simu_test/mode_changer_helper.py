from dataclasses import dataclass
from typing import Dict, Optional, Tuple

@dataclass
class ResilienceDecision:
    action: str                  # "KEEP_MODE" | "SWITCH_MODE" | "ENTER_SAFE_STATE"
    recommended_mode: str        # e.g. "MEC", "PTI", "DP_SAFE"
    reason: str

# -------------------------
# Helper: available capacity
# -------------------------
# def available_mechanical_power(engine_status: Dict[str, bool], me_capacity: float) -> float:
#     return me_capacity if engine_status.get("ME", False) else 0.0

# def available_electrical_power(engine_status: Dict[str, bool], dg_capacity: float) -> float:
#     # simple capacity model: DG1 and DG2 each contribute dg_capacity if available
#     n_dg = int(engine_status.get("DG1", False)) + int(engine_status.get("DG2", False))
#     return n_dg * dg_capacity


def get_mode_capacities(candidate_mode: str, me_capacity: float, dg_capacity: float) -> dict:
    """
    Option B: capacities come from the *candidate mode* (hierarchy-driven),
    not from engine_status DG counts.
    """
    m = candidate_mode.upper()

    if m == "DP_SAFE":
        return {"mech_avail": 0.0, "elec_avail": 0.0, "pto_hotel_via_hsg": False, "req_dg": 0}

    if m.startswith("PTO"):
        # ME provides propulsion + (via HSG generator) hotel load from shaft
        return {"mech_avail": me_capacity, "elec_avail": 0.0, "pto_hotel_via_hsg": True, "req_dg": 0}

    if m == "MEC":
        return {"mech_avail": me_capacity, "elec_avail": 1 * dg_capacity, "pto_hotel_via_hsg": False, "req_dg": 1}

    if m == "MEC_BLACKOUT":
        return {"mech_avail": me_capacity, "elec_avail": 0.0, "pto_hotel_via_hsg": False, "req_dg": 0}

    if m == "PTI":
        return {"mech_avail": 0.0, "elec_avail": 2 * dg_capacity, "pto_hotel_via_hsg": False, "req_dg": 2}

    if m == "PTI_1DG":
        return {"mech_avail": 0.0, "elec_avail": 1 * dg_capacity, "pto_hotel_via_hsg": False, "req_dg": 1}

    # Safe default if you add more PTI variants later
    if m.startswith("PTI"):
        return {"mech_avail": 0.0, "elec_avail": 1 * dg_capacity, "pto_hotel_via_hsg": False, "req_dg": 1}

    raise ValueError(f"Unknown mode tag: {candidate_mode}")




def mode_feasible(
    candidate_mode: str,
    engine_status: Dict[str, bool],
    required_propulsion_power: float,
    required_hotel_power: float,
    me_capacity: float,
    dg_capacity: float,
) -> Tuple[bool, str]:
    """
    Option B: feasibility is evaluated using the *candidate mode* capacities.
    engine_status is used only for hard availability constraints (ME/HSG/DGs broken).
    """
    caps = get_mode_capacities(candidate_mode, me_capacity, dg_capacity)
    mech_avail = caps["mech_avail"]
    elec_avail = caps["elec_avail"]

    # DP_SAFE always feasible as terminal fallback
    if candidate_mode == "DP_SAFE":
        return True, "No-propulsion safe state."

    cm = candidate_mode.upper()

    # Hard constraints: ME required for PTO/MEC families
    if cm.startswith("PTO") or cm.startswith("MEC"):
        if not engine_status.get("ME", False):
            return False, f"{candidate_mode} requires ME."

    # Hard constraints: DG availability required for PTI/MEC electrical (strict)
    # (this ensures hierarchy doesn't choose PTI if DGs are broken)
    req_dg = caps.get("req_dg", 0)
    if req_dg > 0:
        dg_count_alive = int(engine_status.get("DG1", False)) + int(engine_status.get("DG2", False))
        if dg_count_alive < req_dg:
            return False, f"{candidate_mode} requires {req_dg} DG(s), but only {dg_count_alive} available."

    # PTO: hotel supplied via HSG(generator) from shaft power
    if cm.startswith("PTO"):
        if required_hotel_power > 0 and caps["pto_hotel_via_hsg"]:
            if not engine_status.get("HSG", False):
                return False, "PTO requires HSG to supply hotel load."
        required_total_mech = required_propulsion_power + required_hotel_power
        if mech_avail < required_total_mech:
            return False, f"PTO insufficient ME capacity for propulsion+hotel (avail={mech_avail:.0f}, req={required_total_mech:.0f})."
        return True, "PTO feasible: ME covers propulsion + hotel via HSG(generator)."

    # MEC: hotel must be covered by electrical; propulsion by ME
    if cm.startswith("MEC"):
        if elec_avail < required_hotel_power:
            return False, f"MEC insufficient electrical for hotel (avail={elec_avail:.0f}, req={required_hotel_power:.0f})."
        if mech_avail < required_propulsion_power:
            return False, f"MEC insufficient mechanical for propulsion (avail={mech_avail:.0f}, req={required_propulsion_power:.0f})."
        return True, "MEC feasible: ME for propulsion + DG electrical for hotel."

    # PTI: electrical must cover propulsion + hotel
    if cm.startswith("PTI"):
        required_total_elec = required_propulsion_power + required_hotel_power
        if elec_avail < required_total_elec:
            return False, f"{candidate_mode} insufficient electrical (avail={elec_avail:.0f}, req={required_total_elec:.0f})."
        return True, f"{candidate_mode} feasible: electrical covers propulsion + hotel."
    
    print(f"[CHECK] {candidate_mode}: elec_avail={elec_avail:.1f}, "
      f"req_full={required_propulsion_power + required_hotel_power:.1f}, "
      f"req_min={required_hotel_power:.1f}")


    return False, "Unknown mode."


# -------------------------
# Fallback hierarchy per mode
# -------------------------
FALLBACK_HIERARCHY = {
    "PTO": ["PTO", "PTI_1DG", "PTI", "DP_SAFE"],
    "MEC": ["MEC", "PTI_1DG", "PTI", "DP_SAFE"],
    "PTI": ["PTI", "PTO", "MEC_BLACKOUT", "DP_SAFE"],
}

RECOVERY_HIERARCHY = ["MEC", "PTO", "PTI_1DG", "PTI"]


def resilience_supervisor(
    current_mode: str,
    engine_status: Dict[str, bool],
    required_propulsion_power: float,
    required_hotel_power: float,
    me_capacity: float,
    dg_capacity: float,
    min_dwell_s: float,
    time_in_mode_s: float,
) -> ResilienceDecision:
    """
    One-call-per-loop supervisor:
    - checks if current mode is still feasible under faults + power demand
    - if not, walks the hierarchy to find the next feasible mode
    - applies dwell-time to avoid flapping
    """
    # If we're in DP_SAFE, ignore dwell-time and attempt recovery immediately
    if str(current_mode).upper() == "DP_SAFE":
        min_dwell_s = 0.0

    # avoid mode flapping
    if time_in_mode_s < min_dwell_s:
        return ResilienceDecision("KEEP_MODE", current_mode, f"Dwell-time active ({time_in_mode_s:.1f}s < {min_dwell_s:.1f}s).")

    base_mode = current_mode.split("_")[0]  # so "PTI_1DG" maps to "PTI" if you add those later
    hierarchy = FALLBACK_HIERARCHY.get(base_mode, [current_mode, "DP_SAFE"])

    # 1) If current is feasible, keep it
    ok, why = mode_feasible(
        current_mode, engine_status,
        required_propulsion_power, required_hotel_power,
        me_capacity, dg_capacity
    )
    # if ok:
    #     return ResilienceDecision("KEEP_MODE", current_mode, f"Current mode feasible: {why}")

    # 1) Special case: if we're in DP_SAFE, always attempt recovery
    if str(current_mode).upper() == "DP_SAFE":
        for cand in RECOVERY_HIERARCHY:
            ok2, why2 = mode_feasible(
                cand, engine_status,
                required_propulsion_power, required_hotel_power,
                me_capacity, dg_capacity
            )
            if ok2:
                return ResilienceDecision("SWITCH_MODE", cand, f"Recovering from DP_SAFE -> {cand}: {why2}")

        return ResilienceDecision("KEEP_MODE", current_mode, "Remain in DP_SAFE: no propulsive mode feasible.")


    # 2) Else find next feasible candidate
    for cand in hierarchy:
        ok2, why2 = mode_feasible(
            cand, engine_status,
            required_propulsion_power, required_hotel_power,
            me_capacity, dg_capacity
        )
        if ok2:
            if cand == "DP_SAFE":
                return ResilienceDecision("ENTER_SAFE_STATE", cand, f"No propulsive mode feasible; entering safe state: {why2}")
            return ResilienceDecision("SWITCH_MODE", cand, f"Current infeasible ({why}); switching to {cand}: {why2}")

    # 3) Should not happen, but safe fallback
    return ResilienceDecision("ENTER_SAFE_STATE", "DP_SAFE", "No feasible mode found; forcing DP_SAFE.")



def apply_safe_state_overrides(ship_model):
    mach = ship_model.ship_machinery_model
    in_safe = (str(mach.operating_mode).upper() == "DP_SAFE")

    # Store nominal desired speed once (so you don't hardcode 4.0)
    if not hasattr(ship_model, "_desired_speed_nominal"):
        ship_model._desired_speed_nominal = getattr(ship_model, "desired_speed", 0.0)

    # Detect transitions (optional but very useful to avoid integrator windup)
    prev_safe = getattr(ship_model, "_prev_in_safe", False)

    if in_safe:
        # ship_model.turn_off_engine = True

        ship_model.desired_speed = 0.0


        # Optional but recommended: reset throttle controller once when entering DP_SAFE
        if (not prev_safe) and getattr(ship_model, "throttle_controller", None) is not None:
            ship_model.throttle_controller.reset()

    else:
        ship_model.turn_off_engine = False

        # Restore nominal desired speed if you had changed it elsewhere
        ship_model.desired_speed = ship_model._desired_speed_nominal

        # Optional: reset once when leaving DP_SAFE (prevents “surge” due to integrator memory)
        if prev_safe and getattr(ship_model, "throttle_controller", None) is not None:
            ship_model.throttle_controller.reset()

    ship_model._prev_in_safe = in_safe




