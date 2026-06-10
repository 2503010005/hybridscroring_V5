# proxy.py
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import numpy as np

def proxy_fill(df):
    """
    Proxy filling menggunakan XGBRegressor (lebih powerful dari linear)
    """

    mask = df["gen_score"].notna()

    if mask.sum() > 20:

        # =========================
        # FEATURE SET (DITINGKATKAN)
        # =========================
        X = df.loc[mask, ["sim_feat", "rubric_feat", "length_feat"]]
        y = df.loc[mask, "gen_score"]

        model = XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        model.fit(X, y)

        # =========================
        # FILL MISSING
        # =========================
        df.loc[~mask, "gen_score"] = model.predict(
            df.loc[~mask, ["sim_feat", "rubric_feat", "length_feat"]]
        )

        print("✓ Proxy menggunakan XGBRegressor")
        return df, model, "proxy"

    print("⚠ Fallback ke final_score")
    df["gen_score"] = df["final_score"]
    return df, None, "fallback"