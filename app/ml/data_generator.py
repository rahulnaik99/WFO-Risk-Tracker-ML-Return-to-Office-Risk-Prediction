"""
Synthetic employee dataset generator for WFO risk prediction.
Simulates realistic HR data from 2022 post-pandemic return-to-office context.

Features used:
- age: younger employees more comfortable returning
- commute_distance_km: longer commute = higher risk of not returning
- has_children_under_5: caregiving responsibility = higher risk
- vaccination_status: vaccinated = lower risk
- prior_wfo_days_per_week: pre-pandemic office habit
- home_internet_quality: good internet = comfortable WFH = higher risk of staying home
- team_size: larger teams may feel more pressure to return
- manager_wfo: if manager goes to office, employee more likely to follow
- anxiety_score: self-reported (1-10), higher = higher risk
- productivity_wfh_score: self-reported WFH productivity (1-10), higher = prefers WFH
"""

import numpy as np
import pandas as pd


def generate_dataset(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(random_state)

    age = rng.randint(22, 60, n_samples)
    commute_km = rng.exponential(scale=20, size=n_samples).clip(1, 100).round(1)
    has_children = rng.binomial(1, 0.35, n_samples)
    vaccinated = rng.binomial(1, 0.82, n_samples)
    prior_wfo_days = rng.randint(0, 6, n_samples)
    internet_quality = rng.randint(1, 11, n_samples)   # 1=poor, 10=excellent
    team_size = rng.randint(3, 50, n_samples)
    manager_wfo = rng.binomial(1, 0.65, n_samples)
    anxiety_score = rng.randint(1, 11, n_samples)
    productivity_wfh = rng.randint(1, 11, n_samples)

    # Risk score — higher = more likely to NOT return (high WFO risk)
    risk_score = (
        0.20 * (commute_km / 100)
        + 0.15 * has_children
        + 0.10 * (1 - vaccinated)
        + 0.15 * (1 - prior_wfo_days / 5)
        + 0.10 * (internet_quality / 10)
        + 0.10 * (anxiety_score / 10)
        + 0.10 * (productivity_wfh / 10)
        - 0.05 * manager_wfo
        - 0.03 * (age / 60)
        - 0.02 * (team_size / 50)
        + rng.normal(0, 0.05, n_samples)
    ).clip(0, 1)

    # Binary label: 1 = HIGH risk (unlikely to return), 0 = LOW risk
    label = (risk_score > 0.45).astype(int)

    # Risk category
    risk_category = pd.cut(
        risk_score,
        bins=[0, 0.33, 0.66, 1.0],
        labels=["Low", "Medium", "High"]
    )

    df = pd.DataFrame({
        "employee_id": [f"EMP{str(i).zfill(4)}" for i in range(1, n_samples + 1)],
        "age": age,
        "commute_distance_km": commute_km,
        "has_children_under_5": has_children,
        "vaccination_status": vaccinated,
        "prior_wfo_days_per_week": prior_wfo_days,
        "home_internet_quality": internet_quality,
        "team_size": team_size,
        "manager_wfo": manager_wfo,
        "anxiety_score": anxiety_score,
        "productivity_wfh_score": productivity_wfh,
        "risk_score": risk_score.round(3),
        "risk_category": risk_category,
        "wfo_risk_label": label,
    })

    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data/employee_wfo_data.csv", index=False)
    print(f"Dataset saved: {len(df)} rows")
    print(f"Risk distribution:\n{df['risk_category'].value_counts()}")
    print(f"Label balance:\n{df['wfo_risk_label'].value_counts()}")
