"""
Model Verification, Asymmetric Safety Calibration & Physics Ablation Suite (v4)
Project: Flame in Freefall - Predictive Space Habitat Fire Safety System
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from train_dual_ml_engine import (
    CATEGORICAL_FEATURES,
    COMPACT_PHYSICS_NUMERIC,
    RAW_PRE_IGNITION_NUMERIC,
    build_preprocessor,
    engineer_physics_features,
    get_candidate_classifiers,
    get_candidate_regressors,
)


def apply_safety_threshold(probs: np.ndarray, safe_conf_threshold: float = 0.50) -> np.ndarray:
    """Asymmetric ECLSS safety decision rule:
    Only classifies a state as SAFE (0) when P(Safe) >= safe_conf_threshold.
    Otherwise escalates to WARNING (1) or CRITICAL (2) based on relative risk probability.
    """
    preds = np.zeros(len(probs), dtype=int)
    for i, (p0, p1, p2) in enumerate(probs):
        if p0 >= safe_conf_threshold:
            preds[i] = 0
        else:
            preds[i] = 2 if p2 >= p1 * 0.85 else 1
    return preds


def run_validation_suite():
    print("FLAME IN FREEFALL — PEER-REVIEW VALIDATION, SAFETY CALIBRATION & ABLATION (v4)\n")

    clf = joblib.load("models/fire_risk_classifier.pkl")
    reg = joblib.load("models/extinction_time_regressor.pkl")

    raw_df = pd.read_csv("organized_data/master_microgravity_fires.csv")
    feat_df = engineer_physics_features(raw_df)
    phys_cols = COMPACT_PHYSICS_NUMERIC + CATEGORICAL_FEATURES
    raw_cols = RAW_PRE_IGNITION_NUMERIC + CATEGORICAL_FEATURES

    flame_df = feat_df[feat_df["fuel_category"] != "Wire Insulation"].reset_index(drop=True)
    y_cls = flame_df["fire_risk_level"].astype(int).to_numpy()
    groups = flame_df["campaign_group_id"]
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. Compute full Out-of-Fold probabilities using best classifier architecture
    oof_probs = np.zeros((len(flame_df), 3), dtype=float)
    for tr_i, va_i in sgkf.split(flame_df[phys_cols], y_cls, groups):
        pipe = Pipeline([("prep", build_preprocessor(phys_cols)), ("clf", get_candidate_classifiers()["RandomForest"])])
        pipe.fit(flame_df[phys_cols].iloc[tr_i], y_cls[tr_i])
        oof_probs[va_i] = pipe.predict_proba(flame_df[phys_cols].iloc[va_i])

    # 2. Asymmetric Safety Threshold Tradeoff Table (Key Aerospace Systems Figure)
    print("[CHECK 1] Asymmetric ECLSS Safety Threshold Calibration (Full OOF N=403):")
    for tau in [0.33, 0.45, 0.55, 0.65]:
        preds_tau = apply_safety_threshold(oof_probs, safe_conf_threshold=tau)
        cm = confusion_matrix(y_cls, preds_tau)
        cat_misses = cm[2, 0]
        false_alarms = cm[0, 1] + cm[0, 2]
        acc = accuracy_score(y_cls, preds_tau)
        print(
            f"  * Safe Threshold tau={tau:.2f} -> OOF Acc: {acc*100:5.2f}% | "
            f"Catastrophic Misses (Crit->Safe): {cat_misses:2d}/{cm[2].sum()} ({cat_misses/cm[2].sum()*100:4.1f}%) | "
            f"Safe False Escalations: {false_alarms:2d}/{cm[0].sum()}"
        )

    # Per-Investigation OOF Breakdown at standard argmax
    flame_df["oof_pred"] = np.argmax(oof_probs, axis=1)
    print("\n[CHECK 2] Out-of-Fold Accuracy by NASA Investigation (With Sample Thickness + Flow):")
    for ds_name, grp in flame_df.groupby("mission_dataset"):
        print(f"  * {ds_name:<18} (N={len(grp):3d}) -> OOF Accuracy: {accuracy_score(grp['fire_risk_level'], grp['oof_pred'])*100:5.2f}%")

    # 3. Raw Telemetry vs Compact Physics Ablation
    print("\n[CHECK 3] Ablation Study: Raw Telemetry (8 feats) vs. Compact Physics (7 feats):")
    for label, cols in [("Raw Telemetry Only", raw_cols), ("Compact Physics Engine", phys_cols)]:
        oof = np.zeros(len(flame_df), dtype=int)
        for tr_i, va_i in sgkf.split(flame_df[cols], y_cls, groups):
            pipe = Pipeline([("prep", build_preprocessor(cols)), ("clf", get_candidate_classifiers()["RandomForest"])])
            pipe.fit(flame_df[cols].iloc[tr_i], y_cls[tr_i])
            oof[va_i] = pipe.predict(flame_df[cols].iloc[va_i])
        print(f"  * Classification [{label:<22}] -> OOF Acc: {accuracy_score(y_cls, oof)*100:5.2f}% | F1: {f1_score(y_cls, oof, average='weighted'):.4f}")

    ext_df = feat_df[(feat_df["fuel_category"] == "Liquid Droplet") & (feat_df["true_radiative_extinction_s"].notna()) & (feat_df["true_radiative_extinction_s"] <= 120.0)].reset_index(drop=True)
    y_reg = ext_df["true_radiative_extinction_s"].astype(float).to_numpy()
    gkf = GroupKFold(n_splits=5)
    for label, cols in [("Raw Telemetry Only", raw_cols), ("Compact Physics Engine", phys_cols)]:
        oof_r = np.zeros(len(ext_df), dtype=float)
        for tr_i, va_i in gkf.split(ext_df[cols], y_reg, ext_df["campaign_group_id"]):
            pipe = Pipeline([("prep", build_preprocessor(cols)), ("reg", get_candidate_regressors()["RandomForest"])])
            pipe.fit(ext_df[cols].iloc[tr_i], y_reg[tr_i])
            oof_r[va_i] = np.clip(pipe.predict(ext_df[cols].iloc[va_i]), 0.1, None)
        print(f"  * FLEX Regressor [{label:<22}] -> OOF MAE: {mean_absolute_error(y_reg, oof_r):5.3f}s | OOF R2: {r2_score(y_reg, oof_r):.4f}")

    # 4. Domain-Gated Physics Stress Tests
    print("\n[CHECK 4] Domain-Gated Pre-Ignition Physics Stress Tests:")
    scenarios = pd.DataFrame([
        {"scenario": "FLEX Low-O2 Hypoxic Chamber (15% O2)", "pressure_kpa": 101.3, "initial_o2_pct": 15.0, "inert_n2_pct": 84.6, "co2_initial_pct": 0.4, "inert_he_pct": 0.0, "initial_size_mm": 1.6, "sample_thickness_um": np.nan, "flow_velocity_cms": 0.0, "fuel_category": "Liquid Droplet", "fuel_family": "Methanol"},
        {"scenario": "FLEX Enriched O2 Flashover (27% O2)", "pressure_kpa": 101.3, "initial_o2_pct": 27.0, "inert_n2_pct": 72.6, "co2_initial_pct": 0.4, "inert_he_pct": 0.0, "initial_size_mm": 3.5, "sample_thickness_um": np.nan, "flow_velocity_cms": 0.0, "fuel_category": "Liquid Droplet", "fuel_family": "Heptane"},
        {"scenario": "BASS-II Thermally-Thin 100um PMMA (22% O2, 10cm/s)", "pressure_kpa": 101.3, "initial_o2_pct": 22.0, "inert_n2_pct": 77.6, "co2_initial_pct": 0.4, "inert_he_pct": 0.0, "initial_size_mm": 20.0, "sample_thickness_um": 100.0, "flow_velocity_cms": 10.0, "fuel_category": "Solid Polymer/Fabric", "fuel_family": "PMMA_Acrylic"},
        {"scenario": "BASS-II Thermally-Thick 5mm PMMA (22% O2, 10cm/s)", "pressure_kpa": 101.3, "initial_o2_pct": 22.0, "inert_n2_pct": 77.6, "co2_initial_pct": 0.4, "inert_he_pct": 0.0, "initial_size_mm": 20.0, "sample_thickness_um": 5000.0, "flow_velocity_cms": 10.0, "fuel_category": "Solid Polymer/Fabric", "fuel_family": "PMMA_Acrylic"},
    ])
    scen_x = engineer_physics_features(scenarios)[phys_cols]
    probs = clf.predict_proba(scen_x)
    preds = apply_safety_threshold(probs, safe_conf_threshold=0.50)
    risk_names = {0: "SAFE", 1: "WARNING", 2: "CRITICAL"}

    for i, row in scenarios.iterrows():
        lbl = risk_names[int(preds[i])]
        if row["fuel_category"] == "Liquid Droplet":
            t_str = f"{float(np.clip(reg.predict(scen_x.iloc[[i]])[0], 0.1, None)):5.2f}s"
        else:
            t_str = "N/A (Gated: Droplet Phase Only)"
        print(f"  * {row['scenario']:<50} -> {lbl:<8} (P_safe={probs[i][0]*100:4.1f}%) | Ext: {t_str}")


if __name__ == "__main__":
    run_validation_suite()