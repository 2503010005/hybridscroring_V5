from sklearn.ensemble import RandomForestRegressor
import numpy as np

def proxy_fill(df):
    """
    Proxy filling menggunakan RandomForest (lebih powerful dari linear)
    """

    mask = df["gen_score"].notna()

    if mask.sum() > 20:

        # =========================
        # FEATURE SET (DITINGKATKAN)
        # =========================
        X = df.loc[mask, ["sim_feat", "rubric_feat", "length_feat"]]
        y = df.loc[mask, "gen_score"]

        model = RandomForestRegressor(
            n_estimators=150,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X, y)

        # =========================
        # FILL MISSING
        # =========================
        df.loc[~mask, "gen_score"] = model.predict(
            df.loc[~mask, ["sim_feat", "rubric_feat", "length_feat"]]
        )

        print("✓ Proxy menggunakan RandomForest")
        return df, model, "proxy"

    print("⚠ Fallback ke final_score")
    df["gen_score"] = df["final_score"]
    return df, None, "fallback"