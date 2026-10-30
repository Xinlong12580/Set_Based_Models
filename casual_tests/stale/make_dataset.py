import torch
print("running")
src = "../../NN/aidascoutrepo/data/collide_1m_ptrawv1_capped_test.pt"
dst = "data/test.pt"

data = torch.load(src, map_location="cpu", mmap=True)
labels = data["label"]

for label in torch.unique(labels): print(label)
indices = torch.cat([
    torch.where(labels == label)[0][:10000]
    for label in torch.unique(labels) if (int(label) == 0 or int(label) == 2)
])[:20000]

subset = {}
for key, value in data.items():
    if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == labels.shape[0]:
        subset[key] = value[indices].clone()
    else:
        subset[key] = value

torch.save(subset, dst)
print(f"Saved {len(indices)} events to {dst}")
print(subset["label"].long())
print("Class counts:", torch.bincount(subset["label"].long()).tolist())
