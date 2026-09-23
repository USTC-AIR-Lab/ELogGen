"""Canonical environment-interface names used by the ElogGen runtime."""

from __future__ import annotations


LEGACY_ENV_INTERFACE_ALIASES = {
    "MG_OpenArmDrawerStorage": "EG_OpenArmDrawerStorageInterface",
    "MG_OpenArmDrawerStorageInterface": "EG_OpenArmDrawerStorageInterface",
    "MG_OpenArmFruitBasketBagging": "EG_OpenArmFruitBasketBaggingInterface",
    "MG_OpenArmFruitBasketBaggingInterface": "EG_OpenArmFruitBasketBaggingInterface",
    "MG_OpenArmRealExp1": "EG_OpenArmRealExp1Interface",
    "MG_OpenArmRealExp1Interface": "EG_OpenArmRealExp1Interface",
}


def canonicalize_env_interface_name(name):
    """Return the canonical ``EG_`` interface name.

    Historical processed HDF5 files may still contain the old ``MG_`` class
    names in ``datagen_info.attrs['env_interface_name']``. Runtime code should
    treat those values as compatibility aliases only; all newly written metadata
    and all interface lookup paths use the ``EG_`` names.
    """
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    name = str(name)

    if name in LEGACY_ENV_INTERFACE_ALIASES:
        return LEGACY_ENV_INTERFACE_ALIASES[name]

    if name.startswith("MG_"):
        candidate = "EG_" + name[len("MG_") :]
        if not candidate.endswith("Interface"):
            candidate += "Interface"
        return candidate

    return name


def get_canonical_env_interface_info_from_dataset(dataset_path, demo_keys):
    """Read interface metadata after canonicalizing legacy aliases.

    Canonicalization happens *before* the consistency check. This allows a
    historical processed HDF5 that mixes equivalent ``MG_*`` and ``EG_*``
    spellings across episodes to be consumed without propagating the legacy
    name into the runtime.
    """
    import h5py

    env_interface_names = []
    env_interface_types = []
    with h5py.File(dataset_path, "r") as stream:
        for ep in demo_keys:
            datagen_info_key = f"data/{ep}/datagen_info"
            if datagen_info_key not in stream:
                raise AssertionError(
                    "Could not find generation metadata in dataset {}. Ensure you have run "
                    "eloggen prepare on this hdf5".format(dataset_path)
                )
            attrs = stream[datagen_info_key].attrs
            env_interface_names.append(
                canonicalize_env_interface_name(attrs["env_interface_name"])
            )
            interface_type = attrs["env_interface_type"]
            if isinstance(interface_type, bytes):
                interface_type = interface_type.decode("utf-8")
            env_interface_types.append(str(interface_type))

    if not env_interface_names:
        raise ValueError(f"No demos supplied when reading interface metadata from {dataset_path}")

    env_interface_name = env_interface_names[0]
    env_interface_type = env_interface_types[0]
    if not all(elem == env_interface_name for elem in env_interface_names):
        raise AssertionError(
            "Inconsistent canonical environment interface names in source dataset: {}".format(
                sorted(set(env_interface_names))
            )
        )
    if not all(elem == env_interface_type for elem in env_interface_types):
        raise AssertionError(
            "Inconsistent environment interface types in source dataset: {}".format(
                sorted(set(env_interface_types))
            )
        )
    return env_interface_name, env_interface_type


def install_dataset_interface_metadata_reader():
    """Install the canonical metadata reader into the inherited dataset utilities.

    ``generation_runtime.datasets`` is inherited runtime code and still compares
    raw HDF5 attributes. Centralizing the compatibility shim here keeps legacy
    ``MG_*`` handling out of the rest of the generation pipeline while preserving
    the module's public API for both pipeline and direct-runtime entry points.
    """
    import eloggen.generation_runtime.datasets as dataset_utils

    dataset_utils.get_env_interface_info_from_dataset = get_canonical_env_interface_info_from_dataset
