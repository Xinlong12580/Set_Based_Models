"""Read menus and threshold profiles written before the generic menu format."""

from copy import deepcopy
import json

from .presets import PHASE2_MENU


def upgrade_menu(value):
    old_menu = value.get("format") == "collide2v_trigger_menu" and value.get("schema_version") == 1
    old_profile = value.get("format") in (None, "collide2v_pt_threshold_profile") and "thresholds" in value
    if not (old_menu or old_profile):
        raise ValueError("Unsupported saved-menu format/version.")
    result = json.loads(PHASE2_MENU.read_text())
    result["triggers"].extend(deepcopy(value.get("erl_triggers", [])))
    result["object_selection"].update(value.get("object_selection", {}))
    for trigger in result["triggers"]:
        name = trigger["name"]
        trigger["thresholds"] = value.get("thresholds", {}).get(name, trigger["default_thresholds"])
        trigger["target_rate_khz"] = value.get("target_rates_khz", {}).get(name, trigger.get("target_rate_khz"))
    return result
