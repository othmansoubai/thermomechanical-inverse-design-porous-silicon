#!/usr/bin/env python3
# Minimal builder for the INV1 inverse-design geometry.
# Reuses build_structures.py (already here) so INV1 is consistent with the other 18.
import os
import build_structures as bs

# INV1_AR2.5_s37 : Pore A x[0,5] y[0,2], Pore B x[3,8] y[3,5]  (offset +3 = 37.5% stagger, AR=2.5)
bulk  = bs.build_bulk_lattice()
atoms = bs.carve_pores(bulk, 0, 5, 0, 2, 3, 8, 3, 5)

# Output directory: override with STRUCTURES_DIR, else ./structures_for_ovito
out_dir = os.path.expanduser(
    os.environ.get("STRUCTURES_DIR", "structures_for_ovito"))
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "structure_INV1_AR2.5_s37.data")
bs.write_lammps_data(atoms, out, "INV1_AR2.5_s37")

phi = 100.0 * (1.0 - len(atoms) / len(bulk))
print(f"wrote {out}")
print(f"  {len(atoms)} atoms of {len(bulk)}  ->  phi = {phi:.2f}%")
