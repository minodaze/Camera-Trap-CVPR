import pickle
import os

# Simple check
file_path = '/users/PAS2119/hou/ICICLE/ICICLE-Benchmark/log/pipeline/ENO_E06/ce/zs/bioclip_2025-07-03-12-12-12_common_name/full/log/ckp_1_preds.pkl'

print("Checking file:", file_path)
print("File exists:", os.path.exists(file_path))

try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print("Data loaded successfully")
    print("Type:", type(data))
    print("Length:", len(data))
    
    if len(data) == 2:
        preds, labels = data
        print("Predictions shape:", preds.shape)
        print("Labels shape:", labels.shape)
        print("Predictions sample:", preds[:5])
        print("Labels sample:", labels[:5])
        
        # Check if all predictions and labels are the same (might indicate overfitting or error)
        print("Unique predictions:", len(set(preds)))
        print("Unique labels:", len(set(labels)))
        
except Exception as e:
    print("Error:", e)
