"""
Compact Physics-Informed Feature Engineering & Multi-Algorithm Benchmark (v4)
Project: Flame in Freefall - Predictive Space Habitat Fire Safety System
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


RAW_PRE_IGNITION_NUMERIC = [
    "pressure_kpa",
    "initial_o2_pct",
    "inert_n2_pct",
    "co2_initial_pct",
    "inert_he_pct",
    "initial_size_mm",
    "sample_thickness_um",
    "flow_velocity_cms",
]

COMPACT_PHYSICS_NUMERIC = [
    "o2_partial_pressure_kpa",
    "gas_thermal_conductivity",
    "o2_to_inert_ratio",
    "initial_size_mm",
    "droplet_d2_timescale_proxy",          # Quiescent droplet D^2 / P_O2 (FLEX)
    "solid_thermally_thin_spread_proxy",   # Forced-flow inverse-thickness (V_flow * P_O2) / tau_um (BASS-II)
    "radiative_quenching_number",
]

NUMERIC_FEATURES = COMPACT_PHYSICS_NUMERIC
CATEGORICAL_FEATURES = ["fuel_category", "fuel_family"]


def engineer_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "flow_velocity_cms" not in d.columns:
        d["flow_velocity_cms"] = 0.0
    if "sample_thickness_um" not in d.columns:
        d["sample_thickness_um"] = np.where(d["fuel_category"] == "Solid Polymer/Fabric", 400.0, np.nan)

    d["o2_partial_pressure_kpa"] = d["pressure_kpa"] * (d["initial_o2_pct"] / 100.0)
    d["gas_thermal_conductivity"] = (
        (d["inert_he_pct"] / 100.0) * 0.151
        + (d["inert_n2_pct"] / 100.0) * 0.026
        + (d["co2_initial_pct"] / 100.0) * 0.0166
        + (d["initial_o2_pct"] / 100.0) * 0.0265
    )

    inert_sum = (d["inert_n2_pct"] + d["inert_he_pct"] + d["co2_initial_pct"]).replace(0, 1.0)
    d["o2_to_inert_ratio"] = d["initial_o2_pct"] / inert_sum

    is_droplet = d["fuel_category"] == "Liquid Droplet"
    is_solid = d["fuel_category"] == "Solid Polymer/Fabric"

    d["droplet_d2_timescale_proxy"] = np.where(
        is_droplet,
        (d["initial_size_mm"] ** 2) / d["o2_partial_pressure_kpa"].replace(0, 1.0),
        np.nan,
    )

    d["solid_thermally_thin_spread_proxy"] = np.where(
        is_solid,
        (d["flow_velocity_cms"] + 0.5) * d["o2_partial_pressure_kpa"] / (d["sample_thickness_um"].clip(lower=50.0) / 100.0),
        np.nan,
    )

    d["radiative_quenching_number"] = (
        d["gas_thermal_conductivity"] * (1.0 + d["co2_initial_pct"] / 10.0)
    ) / d["o2_to_inert_ratio"].replace(0, 0.01)

    return d


def build_preprocessor(feature_cols: list[str], impute_for_linear: bool = False) -> ColumnTransformer:
    num_cols = [c for c in feature_cols if c not in CATEGORICAL_FEATURES]
    cat_cols = [c for c in feature_cols if c in CATEGORICAL_FEATURES]
    num_transformer = (
        Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
        if impute_for_linear
        else "passthrough"
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )


def get_candidate_classifiers() -> dict:
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=250, max_depth=6, min_samples_leaf=3, random_state=42),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=250, max_depth=7, min_samples_leaf=2, random_state=42),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=140, max_depth=3, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0,
            random_state=42, eval_metric="mlogloss",
        )
    return models


def get_candidate_regressors() -> dict:
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=250, max_depth=6, min_samples_leaf=3, random_state=42),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=250, max_depth=7, min_samples_leaf=2, random_state=42),
        "RidgeLinear": Ridge(alpha=5.0),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBRegressor(
            n_estimators=140, max_depth=3, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0, random_state=42,
        )
    return models


def train_and_export_models(
    data_path: str = "organized_data/master_microgravity_fires.csv",
    model_dir: str = "models",
) -> dict:
    os.makedirs(model_dir, exist_ok=True)
    raw_df = pd.read_csv(data_path)
    feat_df = engineer_physics_features(raw_df)
    all_features = COMPACT_PHYSICS_NUMERIC + CATEGORICAL_FEATURES

    flame_df = feat_df[feat_df["fuel_category"] != "Wire Insulation"].reset_index(drop=True)
    X_cls = flame_df[all_features]
    y_cls = flame_df["fire_risk_level"].astype(int).to_numpy()
    groups_cls = flame_df["campaign_group_id"]

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    print(f"[ENGINE 1 BENCHMARK] Flaming Combustion Classifier (N={len(flame_df)}, 5-Fold StratifiedGroupKFold):")
    best_cls_name, best_cls_acc = "", -1.0

    for name, est in get_candidate_classifiers().items():
        oof = np.zeros(len(flame_df), dtype=int)
        fold_accs = []
        for tr_i, va_i in sgkf.split(X_cls, y_cls, groups_cls):
            pipe = Pipeline([("prep", build_preprocessor(all_features)), ("clf", est)])
            pipe.fit(X_cls.iloc[tr_i], y_cls[tr_i])
            preds = pipe.predict(X_cls.iloc[va_i])
            oof[va_i] = preds
            fold_accs.append(accuracy_score(y_cls[va_i], preds))
        acc = accuracy_score(y_cls, oof)
        f1 = f1_score(y_cls, oof, average="weighted")
        print(f"  * {name:<14} -> Fold Acc: {np.mean(fold_accs)*100:5.2f}% (+/-{np.std(fold_accs)*100:4.2f}%) | OOF Acc: {acc*100:5.2f}% | OOF F1: {f1:.4f}")
        if acc > best_cls_acc:
            best_cls_acc, best_cls_name = acc, name

    clf_pipeline = Pipeline([("prep", build_preprocessor(all_features)), ("clf", get_candidate_classifiers()[best_cls_name])])
    clf_pipeline.fit(X_cls, y_cls)

    # Engine 2: Pure Droplet Radiative Extinction Regressor (exclude all-NaN solid-only feature)
    reg_features = [c for c in all_features if c != "solid_thermally_thin_spread_proxy"]
    ext_df = feat_df[
        (feat_df["fuel_category"] == "Liquid Droplet")
        & (feat_df["true_radiative_extinction_s"].notna())
        & (feat_df["true_radiative_extinction_s"] <= 120.0)
    ].reset_index(drop=True)
    X_reg = ext_df[reg_features]
    y_reg = ext_df["true_radiative_extinction_s"].astype(float).to_numpy()
    gkf = GroupKFold(n_splits=5)

    print(f"\n[ENGINE 2 BENCHMARK] Pure Droplet Radiative Extinction Regressor (FLEX Only, N={len(ext_df)}):")
    best_reg_name, best_reg_r2, best_reg_mae = "", -999.0, 999.0

    for name, est in get_candidate_regressors().items():
        oof_r = np.zeros(len(ext_df), dtype=float)
        is_lin = name == "RidgeLinear"
        for tr_i, va_i in gkf.split(X_reg, y_reg, ext_df["campaign_group_id"]):
            pipe = Pipeline([("prep", build_preprocessor(reg_features, impute_for_linear=is_lin)), ("reg", est)])
            pipe.fit(X_reg.iloc[tr_i], y_reg[tr_i])
            oof_r[va_i] = np.clip(pipe.predict(X_reg.iloc[va_i]), 0.1, None)
        r2 = r2_score(y_reg, oof_r)
        mae = mean_absolute_error(y_reg, oof_r)
        print(f"  * {name:<14} -> OOF MAE: {mae:5.3f} s | OOF R2: {r2:.4f}")
        if r2 > best_reg_r2:
            best_reg_r2, best_reg_mae, best_reg_name = r2, mae, name

    reg_pipeline = Pipeline([
        ("prep", build_preprocessor(reg_features, impute_for_linear=(best_reg_name == "RidgeLinear"))),
        ("reg", get_candidate_regressors()[best_reg_name]),
    ])
    reg_pipeline.fit(X_reg, y_reg)

    joblib.dump(clf_pipeline, os.path.join(model_dir, "fire_risk_classifier.pkl"))
    joblib.dump(reg_pipeline, os.path.join(model_dir, "extinction_time_regressor.pkl"))

    summary = {
        "best_classifier": best_cls_name,
        "classifier_oof_accuracy": round(float(best_cls_acc), 4),
        "best_regressor": best_reg_name,
        "regressor_oof_mae_seconds": round(float(best_reg_mae), 3),
        "regressor_oof_r2": round(float(best_reg_r2), 4),
        "numeric_features": COMPACT_PHYSICS_NUMERIC,
        "regressor_features": reg_features,
        "categorical_features": CATEGORICAL_FEATURES,
    }
    with open(os.path.join(model_dir, "model_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[+] Exported Best Models: Classifier={best_cls_name} ({best_cls_acc*100:.2f}%), Regressor={best_reg_name} (R2={best_reg_r2:.4f})")
    return summary


if __name__ == "__main__":
    train_and_export_models()