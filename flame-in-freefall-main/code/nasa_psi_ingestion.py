"""
Real NASA PSI Archive Organizer & Physical Harmonization Engine (v4 — Thickness & Flow Audited)
Project: Flame in Freefall - Predictive Space Habitat Fire Safety System
"""

import os
import re
import numpy as np
import pandas as pd

BASS2_FAN_CAL_TABLES = {
    0: {
        "volts": np.array([0.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
        "v_cms": np.array([0.0, 3.2, 5.8, 8.5, 11.4, 17.5, 23.8, 30.2, 36.6, 43.0, 49.2, 55.0]),
    },
    1: {
        "volts": np.array([0.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
        "v_cms": np.array([0.0, 1.8, 3.4, 5.1, 6.9, 10.5, 14.2, 18.0, 21.8, 25.6, 29.3, 33.0]),
    },
    2: {
        "volts": np.array([0.0, 1.5, 2.0, 2.5, 3.0, 3.6, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
        "v_cms": np.array([0.0, 0.8, 1.5, 2.2, 2.9, 3.6, 4.1, 5.3, 6.5, 7.7, 8.8, 9.9, 11.0]),
    },
}

BASS2_POT_TO_VOLTS = {
    "pot": np.array([0.0, 1.0, 1.4, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
    "volts": np.array([0.0, 1.6, 2.2, 3.2, 4.8, 6.4, 8.0, 9.2, 10.0, 10.5, 11.0, 11.5]),
}


def clean_numeric_sensor(val, over_range_val: float = 550.0) -> float:
    if pd.isna(val):
        return np.nan
    s = str(val).strip().replace("~", "").replace("\x96", "").replace("–", "").replace("—", "")
    if s == "" or s.lower() in {"none", "nan", "na", "null"}:
        return np.nan
    if s.startswith(">"):
        try:
            return float(s[1:].strip())
        except ValueError:
            return over_range_val
    if s.lower() in {"or", "over"}:
        return over_range_val
    try:
        return float(s)
    except ValueError:
        return np.nan


def sum_numeric_tokens(val) -> float:
    if pd.isna(val):
        return np.nan
    tokens = re.findall(r"\d+\.?\d*", str(val))
    return float(sum(float(t) for t in tokens)) if tokens else np.nan


def parse_bass2_air_display_cms(val) -> float:
    """Extracts true initial flow velocity (cm/s) and removes the +44 anemometer offset on O5b/O13a."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().replace("~", "").replace("\x96", " ")
    if s == "" or s.lower() in {"none", "nan", "na", "null"}:
        return np.nan

    paren_match = re.search(r"\(([^)]+)\)", s)
    if paren_match:
        inner_nums = re.findall(r"[-+]?\d*\.?\d+", paren_match.group(1))
        if inner_nums:
            return float(inner_nums[0])

    cms_match = re.search(r"(\d+\.?\d*)\s*cm/s", s, re.IGNORECASE)
    if cms_match:
        v = float(cms_match.group(1))
        return v - 44.0 if v >= 44.0 else v

    nums = re.findall(r"[-+]?\d*\.?\d+", s)
    if not nums:
        return np.nan
    first_val = float(nums[0])
    if first_val >= 44.0:
        return round(first_val - 44.0, 2)
    return first_val


def parse_bass2_fan_to_cms(fan_val, restrictors: int = 2) -> float:
    if pd.isna(fan_val):
        return np.nan
    s = str(fan_val).strip().replace("~", "").replace("\x96", " ")
    if s == "" or s.lower() in {"none", "nan", "na", "null"}:
        return np.nan

    cal = BASS2_FAN_CAL_TABLES.get(restrictors, BASS2_FAN_CAL_TABLES[2])
    pot_match = re.search(r"pots?\s*(\d*\.?\d+)", s, re.IGNORECASE)
    if pot_match:
        pot_setting = float(pot_match.group(1))
        volts = float(np.interp(pot_setting, BASS2_POT_TO_VOLTS["pot"], BASS2_POT_TO_VOLTS["volts"]))
        return float(np.interp(volts, cal["volts"], cal["v_cms"]))

    tokens = re.findall(r"\d*\.\d+|\d+", s)
    if not tokens:
        return np.nan

    first_tok, first_num = tokens[0], float(tokens[0])
    if "." not in first_tok and 1.0 <= first_num <= 10.0 and len(tokens) >= 2 and tokens[1].startswith("."):
        volts = float(np.interp(first_num, BASS2_POT_TO_VOLTS["pot"], BASS2_POT_TO_VOLTS["volts"]))
        return float(np.interp(volts, cal["volts"], cal["v_cms"]))

    volts = first_num * 10.0 if (first_tok.startswith(".") and first_num < 1.0) else first_num
    return float(np.interp(volts, cal["volts"], cal["v_cms"]))


def extract_solid_thickness_um(material_str: str) -> float:
    """Extracts fuel sample thickness in micrometers (um) for thermally-thin vs thermally-thick scaling:
    - '100 micron PMMA film' -> 100.0 um
    - '300 micron PMMA film' -> 300.0 um
    - '5 mm thick PMMA' -> 5000.0 um
    - '3.2 mm rod' -> 3200.0 um
    - 'SIBAL' / 'Nomex' fabrics -> ~350.0 um (thermally thin woven fabric)
    """
    s = str(material_str).lower()
    m_um = re.search(r"(\d+(?:\.\d+)?)\s*(?:micron|um|µm)", s)
    if m_um:
        return float(m_um.group(1))
    m_mm = re.search(r"(\d+(?:\.\d+)?)\s*mm", s)
    if m_mm:
        return float(m_mm.group(1)) * 1000.0
    if "thick" in s:
        return 5000.0
    if "rod" in s:
        return 3200.0
    if "sibal" in s or "nomex" in s or "cotton" in s:
        return 350.0
    return 400.0


def classify_fuel_family(material_str: str) -> str:
    s = str(material_str).lower()
    if "methanol" in s:
        return "Methanol"
    if "heptane" in s:
        return "Heptane"
    if "rod" in s:
        return "Acrylic_Rod"
    if "pmma" in s or "film" in s or "thick" in s:
        return "PMMA_Acrylic"
    if "sibal" in s:
        return "SIBAL_Cotton_Fiberglass"
    if "nomex" in s:
        return "Nomex_Fabric"
    return "Wire_Polymer"


def parse_flex_psi69(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, encoding="latin1")
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame()
    out["experiment_id"] = "FLEX_" + df["FLEX Identifier"].astype(str).str.strip()
    out["mission_dataset"] = "FLEX (PSI-69)"
    out["fuel_material"] = df["Fuel"].astype(str).str.strip()
    out["fuel_category"] = "Liquid Droplet"

    pressure_kpa = (df["Ambient pressure; mmHg"].apply(clean_numeric_sensor) * 0.133322).where(lambda x: x > 5.0, np.nan)
    out["pressure_kpa"] = pressure_kpa.fillna(pressure_kpa.median())
    out["initial_o2_pct"] = df["O initial ambient composition; mole fraction"].apply(clean_numeric_sensor) * 100.0
    out["inert_n2_pct"] = df["N initial ambient composition; mole fraction"].apply(clean_numeric_sensor) * 100.0
    out["co2_initial_pct"] = df["CO initial ambient composition; mole fraction"].apply(clean_numeric_sensor) * 100.0
    out["inert_he_pct"] = df["He initial ambient composition; mole fraction"].apply(clean_numeric_sensor) * 100.0

    out["initial_size_mm"] = df["Droplet initial diameter; mm"].apply(clean_numeric_sensor)
    out["sample_thickness_um"] = np.nan  # Undefined for liquid droplets
    out["flow_velocity_cms"] = 0.0

    out["obs_burning_rate_mm2_s"] = df["Burning rate; mm"].apply(clean_numeric_sensor)
    out["obs_burn_duration_s"] = df["Burn time; s"].apply(clean_numeric_sensor)
    out["termination_mode"] = df["Test end"].astype(str).str.strip()
    out["true_radiative_extinction_s"] = np.where(
        out["termination_mode"].str.lower() == "extinction", out["obs_burn_duration_s"], np.nan
    )
    out["obs_delta_o2_pct"] = np.nan
    out["obs_delta_co_ppm"] = np.nan
    out["obs_mass_loss_mg_min"] = np.nan
    return out


def parse_bass2_psi25(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, encoding="latin1")
    df.columns = [re.sub(r"\s+", " ", c.strip()) for c in df.columns]
    out = pd.DataFrame()
    out["experiment_id"] = "BASS2_" + df["Test #"].astype(str).str.strip()
    out["mission_dataset"] = "BASS-II (PSI-25)"
    out["fuel_material"] = df["Fuel Sample Material"].astype(str).str.strip()
    out["fuel_category"] = "Solid Polymer/Fabric"

    out["pressure_kpa"] = 101.325
    out["initial_o2_pct"] = df["Calibrated initial O2 % by vol"].apply(clean_numeric_sensor)
    out["inert_n2_pct"] = 100.0 - out["initial_o2_pct"]
    out["co2_initial_pct"] = df["Initial CO2 % by vol"].apply(clean_numeric_sensor).fillna(0.4)
    out["inert_he_pct"] = 0.0

    out["initial_size_mm"] = out["fuel_material"].apply(
        lambda t: float(re.search(r"(\d+(?:\.\d+)?)\s*cm", str(t), re.I).group(1)) * 10.0
        if re.search(r"(\d+(?:\.\d+)?)\s*cm", str(t), re.I) else 20.0
    )
    out["sample_thickness_um"] = out["fuel_material"].apply(extract_solid_thickness_um)

    rest_col = next((c for c in df.columns if "restrictor" in c.lower()), None)
    restrictors = df[rest_col].apply(clean_numeric_sensor).fillna(2).astype(int) if rest_col else pd.Series(2, index=df.index)
    air_col = next((c for c in df.columns if "air display" in c.lower()), None)
    fan_col = next((c for c in df.columns if "fan display" in c.lower()), None)

    v_air = df[air_col].apply(parse_bass2_air_display_cms) if air_col else pd.Series(np.nan, index=df.index)
    v_fan = pd.Series([parse_bass2_fan_to_cms(fv, int(r)) for fv, r in zip(df[fan_col], restrictors)], index=df.index) if fan_col else pd.Series(np.nan, index=df.index)
    out["flow_velocity_cms"] = v_air.fillna(v_fan).fillna(0.0)

    final_o2 = df["Calibrated final O2 % by vol"].apply(clean_numeric_sensor)
    out["obs_delta_o2_pct"] = (out["initial_o2_pct"] - final_o2).clip(lower=0.0).fillna(0.0)
    co_init = df["Initial CO (ppm)"].apply(clean_numeric_sensor).clip(lower=0.0)
    co_final = df["Final CO (ppm)"].apply(clean_numeric_sensor).clip(lower=0.0)
    out["obs_delta_co_ppm"] = (co_final - co_init).clip(lower=0.0).fillna(0.0)

    out["obs_burn_duration_s"] = np.nan
    out["true_radiative_extinction_s"] = np.nan
    out["obs_burning_rate_mm2_s"] = np.nan
    out["obs_mass_loss_mg_min"] = np.nan
    out["termination_mode"] = np.where(out["obs_delta_co_ppm"] > 100.0, "Smoldering/CO Spike", "Solid Flame Spread")
    return out


def parse_same_psi102(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, encoding="latin1")
    df.columns = [c.strip() for c in df.columns]
    valid = df.dropna(subset=["Air Flow (cm/s)", "Weight loss rate (mg/min)"]).copy()
    out = pd.DataFrame()
    out["experiment_id"] = "SAME_" + valid["Sample #"].astype(str)
    out["mission_dataset"] = "SAME (PSI-102)"
    out["fuel_material"] = valid["Sample Material"].astype(str).str.strip()
    out["fuel_category"] = "Wire Insulation"
    out["pressure_kpa"] = 101.325
    out["initial_o2_pct"] = 21.0
    out["inert_n2_pct"] = 79.0
    out["co2_initial_pct"] = 0.04
    out["inert_he_pct"] = 0.0
    out["initial_size_mm"] = 2.0
    out["sample_thickness_um"] = 500.0
    out["flow_velocity_cms"] = valid["Air Flow (cm/s)"].apply(clean_numeric_sensor).fillna(0.0)
    out["obs_mass_loss_mg_min"] = valid["Weight loss rate (mg/min)"].apply(clean_numeric_sensor)
    out["obs_burn_duration_s"] = valid["Aging Duration (minutes)"].apply(clean_numeric_sensor) * 60.0
    out["true_radiative_extinction_s"] = np.nan
    out["obs_burning_rate_mm2_s"] = np.nan
    out["obs_delta_o2_pct"] = np.nan
    out["obs_delta_co_ppm"] = np.nan
    out["termination_mode"] = "Thermal Pyrolysis"
    return out


def assign_empirical_post_test_severity(row: pd.Series) -> int:
    cat = row["fuel_category"]
    if cat == "Liquid Droplet":
        term = str(row.get("termination_mode", "")).strip().lower()
        dur = row.get("obs_burn_duration_s", np.nan)
        k_burn = row.get("obs_burning_rate_mm2_s", np.nan)
        if term == "completion":
            return 2 if ((pd.notna(dur) and dur >= 10.0) or (pd.notna(k_burn) and k_burn >= 0.70)) else 1
        if term == "disruption":
            return 2 if ((pd.notna(k_burn) and k_burn >= 0.75) or (pd.notna(dur) and dur >= 15.0)) else 1
        if pd.notna(k_burn) and k_burn >= 0.75:
            return 2
        if pd.notna(dur) and dur >= 10.0:
            return 1
        return 0
    if cat == "Solid Polymer/Fabric":
        d_co = row.get("obs_delta_co_ppm", 0.0)
        d_o2 = row.get("obs_delta_o2_pct", 0.0)
        if (pd.notna(d_co) and d_co >= 100.0) or (pd.notna(d_o2) and d_o2 >= 0.60):
            return 2
        if (pd.notna(d_co) and d_co >= 20.0) or (pd.notna(d_o2) and d_o2 >= 0.15):
            return 1
        return 0
    m_loss = row.get("obs_mass_loss_mg_min", 0.0)
    return 2 if (pd.notna(m_loss) and m_loss >= 1.5) else (1 if (pd.notna(m_loss) and m_loss >= 0.4) else 0)


def build_organized_nasa_dataset(data_dir: str = "data", output_dir: str = "organized_data") -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    search_dirs = [data_dir, "."]

    def locate(filename: str) -> str:
        for d in search_dirs:
            c = os.path.join(d, filename)
            if os.path.exists(c):
                return c
        raise FileNotFoundError(f"Could not locate '{filename}'")

    flex_df = parse_flex_psi69(locate("PSI-69_Experimental table_FLEX.csv"))
    bass_df = parse_bass2_psi25(locate("PSI-25_Experimental table_BASS-II.csv"))
    same_df = parse_same_psi102(locate("PSI-102_Experimental table_SAME.csv"))

    master_df = pd.concat([flex_df, bass_df, same_df], ignore_index=True).dropna(subset=["initial_o2_pct"]).copy()
    master_df["initial_size_mm"] = master_df.groupby("fuel_category")["initial_size_mm"].transform(lambda s: s.fillna(s.median()))
    master_df["fuel_family"] = master_df["fuel_material"].apply(classify_fuel_family)

    o2_bin = (master_df["initial_o2_pct"] / 2.0).round() * 2.0
    p_bin = (master_df["pressure_kpa"] / 15.0).round() * 15.0
    v_bin = (master_df["flow_velocity_cms"] / 5.0).round() * 5.0
    master_df["campaign_group_id"] = (
        master_df["mission_dataset"] + "__" + master_df["fuel_family"]
        + "__O2_" + o2_bin.astype(str) + "__P_" + p_bin.astype(str) + "__V_" + v_bin.astype(str)
    )
    master_df["fire_risk_level"] = master_df.apply(assign_empirical_post_test_severity, axis=1)
    master_df["fire_risk_label"] = master_df["fire_risk_level"].map({0: "Safe", 1: "Warning", 2: "Critical"})

    master_path = os.path.join(output_dir, "master_microgravity_fires.csv")
    master_df.to_csv(master_path, index=False)

    bass_sub = master_df[master_df["fuel_category"] == "Solid Polymer/Fabric"]
    print(f"[SUCCESS] v4 Harmonized N={len(master_df)} | BASS-II Flow Max={bass_sub['flow_velocity_cms'].max():.2f} cm/s | Thicknesses={sorted(bass_sub['sample_thickness_um'].unique())}")
    return master_df


if __name__ == "__main__":
    build_organized_nasa_dataset()