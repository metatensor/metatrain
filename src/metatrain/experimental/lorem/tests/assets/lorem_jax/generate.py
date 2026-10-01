"""Write the lorem-jax reference outputs used by ``test_flax_checkpoint.py``.

Run with lorem-jax installed, from this folder::

    python generate.py

The current files come from lorem-jax 4535c9f (``main`` after 0.1.0, which
adds the long-range contribution to the Born charges, lab-cosmo/lorem-jax#30)
with marathon-train 0.3.1.

Each ``<case>.npz`` holds the flat flax parameters (``params/<flax path>``),
the ``model.yaml`` text (``marathon.io.to_dict`` of the model, as lorem-jax
writes it into checkpoints), and the lorem-jax predictions for two small systems,
in float64. Parameters are the random initialization plus noise of scale
``NOISE``, so every bias and LayerNorm leaf is exercised and the parameters
are far enough from initialization that swapped tensor-product factors
change the energy well above the test tolerance. ``PerParticleTensorPredictor``
is replaced by a version that chains its Dense layers, the fix for
lab-cosmo/lorem-jax#38 (drop it once lab-cosmo/lorem-jax#40 is released).
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
from jaxpme.batched_mixed.batching import prepare
from lorem import Lorem, LoremBEC
from lorem.batching import Sample, to_batch
from marathon.data.sample import to_labels
from marathon.io import to_dict


jax.config.update("jax_enable_x64", True)


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

NOISE = 0.3
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
    "lorem": (Lorem, {}),
    "lorem_scalar_messages": (Lorem, {"equivariant_message_passing": False}),
    "lorem_bec": (LoremBEC, {}),
    # the paper default is max_degree=6; degree 4 covers the l >= 3
    # Clebsch-Gordan signs and a degree-2 long-range charge
    "lorem_l4": (Lorem, {"max_degree": 4, "max_degree_lr": 2}),
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
    for case, (cls, extra) in CASES.items():
        config = {**SIZES, **extra}
        model = cls(**config)
        params = model.init(jax.random.key(0), *model.dummy_inputs())
        leaves, tree = jax.tree.flatten(params)
        keys = jax.random.split(jax.random.key(1), len(leaves))
        leaves = [
            x + NOISE * jax.random.normal(k, x.shape, x.dtype)
            for x, k in zip(leaves, keys, strict=True)
        ]
        params = jax.tree.unflatten(tree, leaves)

        data = {
            f"params/{path}": np.asarray(value)
            for path, value in flatten_dict(params["params"], sep="/").items()
        }
        data["model_yaml"] = np.array(yaml.safe_dump(to_dict(model)))
        for label, atoms in SYSTEMS.items():
            # lorem.batching.to_sample, but in float64 (it hard-codes float32)
            structure = prepare(atoms, config["cutoff"], dtype=np.float64)
            sample = Sample(structure, to_labels(atoms, energy=False, forces=False))
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
