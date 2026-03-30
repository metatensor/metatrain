import ase.io


all_structures = ase.io.read("mad-val-metatrain.xyz", index=":")
new_structures = []
for s in all_structures:
    if not s.pbc.any() or not s.pbc.all():
        continue
    new_structures.append(s)
print("Dropped", len(all_structures) - len(new_structures), "structures")
ase.io.write("mad-val-metatrain-filtered.xyz", new_structures)
