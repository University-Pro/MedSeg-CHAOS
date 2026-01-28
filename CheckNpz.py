import os
import numpy as np

base_dir = "./Datasets/Chaos"
patient_dir = os.path.join(base_dir, "t1", "1")
values = set()
for file in os.listdir(patient_dir):
    if file.endswith("outphase_0.npz"):
        continue  # 可选，只检查 outphase
    data = np.load(os.path.join(patient_dir, file))
    values.update(np.unique(data["lab"]))
print(values)