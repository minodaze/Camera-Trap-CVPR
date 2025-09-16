import pandas as pd

# Load the CSV
df = pd.read_csv('csv/CL + Animal Trap - detailed metrics.csv')

# List of camera names to keep (these are in the second column)
keep = [
    'orinoquia_N25',
    'na_archbold_FL-32',
    'serengeti_E08',
    'serengeti_K13',
    'serengeti_T13',
    'serengeti_F13',
    'serengeti_E03',
    'KGA_KHOGB07',
    'nz_EFH_HCAMC01',
    'KGA_KHOLA08',
    'orinoquia_N29',
    'nz_EFH_HCAMD03',
    'nz_EFH_HCAMB10',
    'KGA_KHOLA03',
    'nz_PS1_CAM6213',
    'nz_EFD_DCAMB02',
]

# Filter the DataFrame to keep only the specified camera names (second column)
selected_df = df[df.iloc[:, 1].isin(keep)]

# Keep all columns with their metric data (not just the first column)
# selected_df already contains all the columns we want

# Save to a new CSV
selected_df.to_csv('csv/filtered_metrics.csv', index=False)