"""Write the lorem-jax reference outputs used by ``test_flax_checkpoint.py``.

Run with lorem-jax installed (``pip install lorem-jax``), from this folder::

    python generate.py

Each ``<case>.npz`` holds the flat flax parameters (``params/<flax path>``),
the ``model.yaml`` text, and the lorem-jax predictions for two small systems.
Parameters are the random initialization plus noise, so every bias and
LayerNorm leaf is exercised. ``PerParticleTensorPredictor`` is replaced by a
version that chains its Dense layers, the fix for lab-cosmo/lorem-jax#38.
"""

import ase
import e3x
import flax.linen as nn
import jax
import jax.numpy as jnp
import lorem.models.bec
import numpy as np
import yaml
from flax.traverse_util import flatten_dict
from lorem import Lorem, LoremBEC
from lorem.batching import to_batch, to_sample


class PerParticleTensorPredictor(nn.Module):
    features: int = 128

    @nn.compact
    def __call__(self, spherical_features):
        x = e3x.nn.silu(e3x.nn.Dense(self.features)(spherical_features))
        x = e3x.nn.silu(e3x.nn.Dense(self.features)(x))
        x = e3x.nn.Dense(self.features)(x)
        x = e3x.nn.TensorDense(features=1, max_degree=2)(x)[..., 0, :, 0]
        cg = e3x.so3.clebsch_gordan(max_degree1=1, max_degree2=1, max_degree3=2)
        return jnp.einsum("...l,nml->...nm", x, cg[1:, 1:, :])


lorem.models.bec.PerParticleTensorPredictor = PerParticleTensorPredictor

SIZES = dict(
    cutoff=3.0,
    max_degree=2,
    max_degree_lr=1,
    num_features=4,
    num_radial=3,
    num_species=4,
    num_spherical_features=2,
    num_message_passing=1,
)
CASES = {
    "lorem": ("lorem.Lorem", Lorem, {}),
    "lorem_scalar_messages": (
        "lorem.Lorem",
        Lorem,
        {"equivariant_message_passing": False},
    ),
    "lorem_bec": ("lorem.LoremBEC", LoremBEC, {}),
}
SYSTEMS = {
    "molecule": ase.Atoms(
        "HCOO",
        positions=[[0.0, 0.0, 0.0], [1.1, 0.2, 0.1], [0.3, 1.2, 0.4], [-0.5, 0.4, 1.1]],
    ),
    "crystal": ase.Atoms(
        "HCOO",
        positions=[[0.1, 0.2, 0.0], [1.3, 0.1, 0.4], [0.2, 1.5, 0.9], [2.1, 1.9, 1.7]],
        cell=[[4.0, 0.0, 0.0], [0.3, 4.2, 0.0], [0.0, 0.5, 4.4]],
        pbc=True,
    ),
}


def main():
    for case, (name, cls, extra) in CASES.items():
        config = {**SIZES, **extra}
        model = cls(**config)
        params = model.init(jax.random.key(0), *model.dummy_inputs())
        leaves, tree = jax.tree.flatten(params)
        keys = jax.random.split(jax.random.key(1), len(leaves))
        leaves = [
            x + 0.1 * jax.random.normal(k, x.shape)
            for x, k in zip(leaves, keys, strict=True)
        ]
        params = jax.tree.unflatten(tree, leaves)

        data = {
            f"params/{path}": np.asarray(value)
            for path, value in flatten_dict(params["params"], sep="/").items()
        }
        data["model_yaml"] = np.array(yaml.safe_dump({"model": {name: config}}))
        for label, atoms in SYSTEMS.items():
            sample = to_sample(atoms, config["cutoff"], energy=False, forces=False)
            batch = jax.tree.map(jnp.asarray, to_batch([sample], []))
            results = model.predict(params, batch)
            n = len(atoms)
            data[f"{label}/numbers"] = atoms.numbers
            data[f"{label}/positions"] = atoms.positions
            data[f"{label}/cell"] = atoms.cell.array
            data[f"{label}/pbc"] = atoms.pbc
            data[f"{label}/energy"] = np.asarray(results["energy"])[0]
            data[f"{label}/forces"] = np.asarray(results["forces"])[:n]
            if "apt" in results:
                data[f"{label}/apt"] = np.asarray(results["apt"])[:n]
        np.savez_compressed(f"{case}.npz", **data)


if __name__ == "__main__":
    main()
