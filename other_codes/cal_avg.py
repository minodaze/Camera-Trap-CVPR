import os
import pandas as pd
import json

zs_list = [94.1 , 97.7 , 81.3 , 94.5 , 75.8 , 81.5 , 85.6 , 80.1 , 81.2 , 62.7 , 79.3 , 75.3 , 69.5 , 87.2 , 82.6 , 76.5 , 92.3 , 77.9 , 80.2 , 82.1]
accum_list = [63.2 , 73.6 , 56.0 , 72.9 , 57.0 , 50.0 , 75.7 , 60.0 , 73.0 , 60.6 , 76.0 , 66.0 , 66.0 , 75.1 , 68.0 , 68.0 , 77.1 , 73.0 , 74.0 , 72.0]
best_accum_list = [98.0 , 98.7 , 79.6 , 94.7 , 76.9 , 69.7 , 94.5 , 75.9 , 88.3 , 75.4 , 90.7 , 80.0 , 79.6 , 88.2 , 81.0 , 80.9 , 89.9 , 84.9 , 85.4 , 83.0]
best_oracle_list = [96.1 , 98.7 , 84.9 , 96.5 , 84.3 , 82.2 , 90.0 , 81.6 , 91.0 , 89.7 , 92.1 , 82.4 , 78.7 , 90.3 , 87.8 , 87.0 , 96.4 , 91.6 , 87.8 , 86.8]

avg_zs = sum(zs_list) / len(zs_list)
avg_accum = sum(accum_list) / len(accum_list)
avg_best_accum = sum(best_accum_list) / len(best_accum_list)
avg_best_oracle = sum(best_oracle_list) / len(best_oracle_list)

print(f"Average zero-shot balanced accuracy: {avg_zs:.2f}")
print(f"Average balanced accuracy at the last checkpoint: {avg_accum:.2f}")
print(f"Average best balanced accuracy across checkpoints: {avg_best_accum:.2f}")
print(f"Average best oracle balanced accuracy across checkpoints: {avg_best_oracle:.2f}")