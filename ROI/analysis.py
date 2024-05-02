import pandas as pd
from scipy import stats

metric = 'r2'
df = pd.read_csv(f'result_csv/{metric}_mean_ROI_41-50.csv')
# Compute overall means for MAE and R^2
mean_DTI = df['DTI'].mean()
mean_FWE = df['FWE'].mean()

# Compute standard errors of the means (SEM)
sem_DTI = df['DTI'].sem()
sem_FWE = df['FWE'].sem()

# Determine the t-critical value for 95% CI, assuming two-tailed and df=n-1
alpha = 0.05
df_DTI = len(df['DTI']) - 1  # degrees of freedom for MAE
df_FWE = len(df['FWE']) - 1  # degrees of freedom for R^2
t_critical_DTI = stats.t.ppf(1 - alpha/2, df_DTI)
t_critical_FWE = stats.t.ppf(1 - alpha/2, df_FWE)

# Compute the 95% confidence intervals
ci_DTI = t_critical_DTI * sem_DTI
ci_FWE = t_critical_FWE * sem_FWE

# Print the results
print(f"DTI = {mean_DTI:.3f} [{mean_DTI-ci_DTI:.3f}, {mean_DTI+ci_DTI:.3f}]")
print(f"FWE = {mean_FWE:.3f} [{mean_FWE-ci_FWE:.3f}, {mean_FWE+ci_FWE:.3f}]")