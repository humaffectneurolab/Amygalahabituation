
"""
Analysis script for within-subject variability
as a function of Intolerance of Uncertainty (IU) group.

"""

import pandas as pd
import numpy as np
from scipy import stats


# 1. Load data and define IU groups (based on IUS z-scores)

file_path = r"C:/path/to/data/file.xlsx"

df = pd.read_excel(file_path)

# Compute subject-level IUS and z-score across subjects
subj_iu = df.groupby("subject", as_index=False)["IU"].first()

z = (subj_iu["IU"] - subj_iu["IU"].mean()) / subj_iu["IU"].std(ddof=1)

# Define IU groups based on z-score thresholds (±1 SD)
subj_iu["IU_group"] = np.where(
    z >= 1.0, "High IU (+1SD)",
    np.where(z <= -1.0, "Low IU (-1SD)", "Mid")
)

# Merge IU_group back to the trial-level dataframe
df = df.merge(subj_iu[["subject", "IU_group"]], on="subject", how="left")

# 2. Compute within-subject SD of beta across blocks

# For each subject (and IU_group), compute within-subject SD of beta across blocks.
sd_per_subject = (
    df.groupby(["subject", "IU_group"])["beta"]
      .agg(within_sd=lambda x: np.std(x, ddof=1))
      .reset_index()
)


# 3. Group-level summary statistics and 95% CI

def mean_ci(series, alpha=0.05):
    data = series.to_numpy()
    n = len(data)
    mean = np.mean(data)
    sd = np.std(data, ddof=1)
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    ci_low = mean - tcrit * se
    ci_high = mean + tcrit * se
    return pd.Series(
        {"mean": mean, "sd": sd, "n": n, "ci_low": ci_low, "ci_high": ci_high}
    ) 
# Apply the summary function to within-subject SDs by IU group
within_sd_summary = (
    sd_per_subject
    .groupby("IU_group")["within_sd"]
    .apply(mean_ci)
    .unstack()  # columns: mean, sd, n, ci_low, ci_high
)

# Clean and round the summary table
summary_table = within_sd_summary[["mean", "sd", "n", "ci_low", "ci_high"]].round(4)


# 4. Welch t-test (High IU vs Low IU) and effect size (Hedges' g)


def welch_test_and_effects(a, b, alpha=0.05):
    # Arrays directly (NaN-free by assumption)
    a = np.asarray(a)
    b = np.asarray(b)

    # Welch t-test
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)

    # Descriptive stats
    n1, n2 = len(a), len(b)
    m1, m2 = np.mean(a), np.mean(b)
    s1, s2 = np.std(a, ddof=1), np.std(b, ddof=1)

    # Standard error + Welch-Satterthwaite df
    se_diff = np.sqrt((s1**2 / n1) + (s2**2 / n2))
    df_welch = (
        (s1**2 / n1 + s2**2 / n2)**2 /
        (((s1**2 / n1)**2) / (n1 - 1) + ((s2**2 / n2)**2) / (n2 - 1))
    )

    # 95% CI of mean difference
    tcrit = stats.t.ppf(1 - alpha/2, df=df_welch)
    diff = m1 - m2
    ci_low = diff - tcrit * se_diff
    ci_high = diff + tcrit * se_diff

    # Hedges' g
    s_pooled = np.sqrt(((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / (n1 + n2 - 2))
    d = diff / s_pooled
    J = 1 - (3 / (4*(n1 + n2) - 9))
    g = d * J

    return {
        "t": t_stat,
        "p": p_val,
        "df_welch": df_welch,
        "mean_diff": diff,
        "diff_ci_low": ci_low,
        "diff_ci_high": ci_high,
        "hedges_g": g,
        "n_high": n1,
        "n_low": n2,
        "mean_high": m1,
        "mean_low": m2
    }

high_sd = sd_per_subject.loc[
    sd_per_subject["IU_group"] == "High IU (+1SD)", "within_sd"
]
low_sd = sd_per_subject.loc[
    sd_per_subject["IU_group"] == "Low IU (-1SD)", "within_sd"
]

welch_res = welch_test_and_effects(high_sd, low_sd)

# 6. Print results

print("Within-subject SD (beta across blocks) by IU group")
print(summary_table.to_string())

print("High vs Low IU: Welch t-test on within-subject SD")
for k, v in welch_res.items():
    if isinstance(v, float):
        print(f"{k:>12s}: {v:.4f}")
    else:
        print(f"{k:>12s}: {v}")
        