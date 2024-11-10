# """Test correct type and metadata of readers. Correct values will be checked
# within the tests for each reader."""

# import logging

# import ase
# import ase.io
# import pytest
# import torch
# from metatensor.torch import Labels
# from omegaconf import OmegaConf
# from test_targets_ase import ase_system, ase_systems

# from metatrain.utils.data.dataset import TargetInfo
# from metatrain.utils.data.readers import (
#     read_energy,
#     read_forces,
#     read_stress,
#     read_systems,
#     read_targets,
#     read_virial,
# )
# from metatrain.utils.data.readers.readers import _base_reader


# @pytest.mark.parametrize("reader", (None, "ase"))
# def test_read_systems(reader, monkeypatch, tmp_path):
#     monkeypatch.chdir(tmp_path)

#     filename = "systems.xyz"
#     systems = ase_systems()
#     ase.io.write(filename, systems)

#     results = read_systems(filename, reader=reader)

#     assert isinstance(results, list)
#     assert len(results) == len(systems)
#     for system, result in zip(systems, results):
#         assert isinstance(result, torch.ScriptObject)

#         torch.testing.assert_close(
#             result.positions, torch.tensor(system.positions, dtype=torch.float64)
#         )
#         torch.testing.assert_close(
#             result.types, torch.tensor([1, 1], dtype=torch.int32)
#         )


# def test_read_systems_unknown_reader():
#     match = "File extension '.bar' is not linked to a default reader"
#     with pytest.raises(ValueError, match=match):
#         read_systems("foo.bar")


# def test_read_unknonw_library():
#     match = "Reader library 'foo' not supported."
#     with pytest.raises(ValueError, match=match):
#         read_systems("foo.foo", reader="foo")


# @pytest.mark.parametrize("reader", (None, "ase"))
# def test_read_energies(reader, monkeypatch, tmp_path):
#     monkeypatch.chdir(tmp_path)

#     filename = "systems.xyz"
#     systems = ase_systems()
#     ase.io.write(filename, systems)

#     results = read_energy(filename, reader=reader, target_value="true_energy")

#     assert type(results) is list
#     assert len(results) == len(systems)
#     for i_system, result in enumerate(results):
#         assert result.values.dtype is torch.float64
#         assert result.samples.names == ["system"]
#         assert result.samples.values == torch.tensor([[i_system]])
#         assert result.properties == Labels("energy", torch.tensor([[0]]))


# @pytest.mark.parametrize("reader", (None, "ase"))
# def test_read_forces(reader, monkeypatch, tmp_path):
#     monkeypatch.chdir(tmp_path)

#     filename = "systems.xyz"
#     systems = ase_systems()
#     ase.io.write(filename, systems)

#     results = read_forces(filename, reader=reader, target_value="forces")

#     assert type(results) is list
#     assert len(results) == len(systems)
#     for i_system, result in enumerate(results):
#         assert result.values.dtype is torch.float64
#         assert result.samples.names == ["sample", "system", "atom"]
#         assert torch.all(result.samples["sample"] == torch.tensor(0))
#         assert torch.all(result.samples["system"] == torch.tensor(i_system))
#         assert result.components == [Labels(["xyz"], torch.arange(3).reshape(-1, 1))]
#         assert result.properties == Labels("energy", torch.tensor([[0]]))


# @pytest.mark.parametrize("reader_func", [read_stress, read_virial])
# @pytest.mark.parametrize("reader", (None, "ase"))
# def test_read_stress_virial(reader_func, reader, monkeypatch, tmp_path):
#     monkeypatch.chdir(tmp_path)

#     filename = "systems.xyz"
#     systems = ase_systems()
#     ase.io.write(filename, systems)

#     results = reader_func(filename, reader=reader, target_value="stress-3x3")

#     assert type(results) is list
#     assert len(results) == len(systems)
#     components = [
#         Labels(["xyz_1"], torch.arange(3).reshape(-1, 1)),
#         Labels(["xyz_2"], torch.arange(3).reshape(-1, 1)),
#     ]
#     for result in results:
#         assert result.values.dtype is torch.float64
#         assert result.samples.names == ["sample"]
#         assert result.samples.values == torch.tensor([[0]])
#         assert result.components == components
#         assert result.properties == Labels("energy", torch.tensor([[0]]))


# @pytest.mark.parametrize(
#     "reader_func", [read_energy, read_forces, read_stress, read_virial]
# )
# def test_reader_unknown_reader(reader_func):
#     match = "File extension '.bar' is not linked to a default reader"
#     with pytest.raises(ValueError, match=match):
#         reader_func("foo.bar", target_value="baz")


# def test_reader_unknown_target():
#     match = "Reader library 'ase' can't read 'mytarget'."
#     with pytest.raises(ValueError, match=match):
#         _base_reader(target="mytarget", filename="structures.xyz", reader="ase")


# STRESS_VIRIAL_DICT = {
#     "read_from": "systems.xyz",
#     "reader": "ase",
#     "key": "stress-3x3",
# }


# @pytest.mark.parametrize(
#     "stress_dict, virial_dict",
#     [[STRESS_VIRIAL_DICT, False], [False, STRESS_VIRIAL_DICT]],
# )
# def test_read_targets(stress_dict, virial_dict, monkeypatch, tmp_path, caplog):
#     monkeypatch.chdir(tmp_path)

#     filename = "systems.xyz"
#     systems = ase_systems()
#     ase.io.write(filename, systems)

#     energy_section = {
#         "quantity": "energy",
#         "read_from": filename,
#         "reader": "ase",
#         "key": "true_energy",
#         "unit": "eV",
#         "type": "scalar",
#         "per_atom": False,
#         "num_properties": 1,
#         "forces": {"read_from": filename, "reader": "ase", "key": "forces"},
#         "stress": stress_dict,
#         "virial": virial_dict,
#     }

#     conf = {
#         "energy": energy_section,
#         "mtt::energy2": energy_section,
#     }

#     caplog.set_level(logging.INFO)
#     result, target_info_dict = read_targets(OmegaConf.create(conf))

#     assert any(["Forces found" in rec.message for rec in caplog.records])

#     assert type(result) is dict
#     assert type(target_info_dict) is dict

#     if stress_dict:
#         assert any(["Stress found" in rec.message for rec in caplog.records])
#     if virial_dict:
#         assert any(["Virial found" in rec.message for rec in caplog.records])

#     for target_name, target_list in result.items():
#         target_section = conf[target_name]

#         target_info = target_info_dict[target_name]
#         assert type(target_info) is TargetInfo
#         assert target_info.quantity == target_section["quantity"]
#         assert target_info.unit == target_section["unit"]
#         assert target_info.per_atom is False
#         assert target_info.gradients == ["positions", "strain"]

#         assert type(target_list) is list
#         for target in target_list:
#             assert target.keys == Labels(["_"], torch.tensor([[0]]))

#             result_block = target.block()
#             assert result_block.values.dtype is torch.float64
#             assert result_block.samples.names == ["system"]
#             assert result_block.properties == Labels("energy", torch.tensor([[0]]))

#             pos_grad = result_block.gradient("positions")
#             assert pos_grad.values.dtype is torch.float64
#             assert pos_grad.samples.names == ["sample", "system", "atom"]
#             assert pos_grad.components == [
#                 Labels(["xyz"], torch.arange(3).reshape(-1, 1))
#             ]
#             assert pos_grad.properties == Labels("energy", torch.tensor([[0]]))

#             strain_grad = result_block.gradient("strain")
#             components = [
#                 Labels(["xyz_1"], torch.arange(3).reshape(-1, 1)),
#                 Labels(["xyz_2"], torch.arange(3).reshape(-1, 1)),
#             ]
#             assert strain_grad.values.dtype is torch.float64
#             assert strain_grad.samples.names == ["sample"]
#             assert strain_grad.components == components
#             assert strain_grad.properties == Labels("energy", torch.tensor([[0]]))


# @pytest.mark.parametrize(
#     "stress_dict, virial_dict",
#     [[STRESS_VIRIAL_DICT, False], [False, STRESS_VIRIAL_DICT]],
# )
# def test_read_targets_warnings(stress_dict, virial_dict, monkeypatch, tmp_path, caplo
#     monkeypatch.chdir(tmp_path)

#     filename = "systems.xyz"
#     systems = ase_system()

#     # Delete gradient sections
#     systems.info.pop("stress-3x3")
#     systems.info.pop("stress-9")
#     systems.arrays.pop("forces")

#     ase.io.write(filename, systems)

#     energy_section = {
#         "quantity": "energy",
#         "read_from": filename,
#         "reader": "ase",
#         "key": "true_energy",
#         "unit": "eV",
#         "type": "scalar",
#         "per_atom": False,
#         "num_properties": 1,
#         "forces": {"read_from": filename, "reader": "ase", "key": "forces"},
#         "stress": stress_dict,
#         "virial": virial_dict,
#     }

#     conf = {"energy": energy_section}

#     caplog.set_level(logging.WARNING)
#     read_targets(OmegaConf.create(conf))  # , slice_samples_by="system")

#     assert any(["No forces found" in rec.message for rec in caplog.records])

#     if stress_dict:
#         assert any(["No stress found" in rec.message for rec in caplog.records])
#     if virial_dict:
#         assert any(["No virial found" in rec.message for rec in caplog.records])


# def test_read_targets_error(monkeypatch, tmp_path):
#     monkeypatch.chdir(tmp_path)

#     filename = "systems.xyz"
#     systems = ase_system()
#     ase.io.write(filename, systems)

#     energy_section = {
#         "quantity": "energy",
#         "read_from": filename,
#         "reader": "ase",
#         "key": "true_energy",
#         "type": "scalar",
#         "per_atom": False,
#         "num_properties": 1,
#         "forces": {"read_from": filename, "reader": "ase", "key": "forces"},
#         "stress": True,
#         "virial": True,
#     }

#     conf = {"energy": energy_section}

#     with pytest.raises(
#         ValueError,
#         match="stress and virial at the same time",
#     ):
#         read_targets(OmegaConf.create(conf))


# def test_unsupported_quantity():
#     conf = {
#         "energy": {
#             "quantity": "foo",
#         }
#     }

#     with pytest.raises(
#         ValueError,
#         match="Quantity: 'foo' is not supported. Choose 'energy'.",
#     ):
#         read_targets(OmegaConf.create(conf))


# def test_unsupported_target_name():
#     conf = {
#         "free_energy": {
#             "quantity": "energy",
#         }
#     }

#     with pytest.raises(
#         ValueError,
#         match="start with `mtt::`",
#     ):
#         read_targets(OmegaConf.create(conf))
