from pathlib import Path
from typing import Any, List, Set, Tuple

from omegaconf import DictConfig, ListConfig, OmegaConf


_IMPORT_KEYS = ("imports", "includes", "include")
_EXTENDS_KEYS = ("extends", "_base_")


def _to_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, ListConfig)):
        return [str(v) for v in value]
    raise TypeError(f"expected str or list[str], got {type(value)}")


def _normalize_import_items(value: Any):
    if value is None:
        return []

    if isinstance(value, str):
        return [(None, value)]

    if isinstance(value, (list, tuple, ListConfig)):
        items = []
        for item in value:
            if isinstance(item, str):
                items.append((None, item))
            elif isinstance(item, (dict, DictConfig)):
                for target_path, import_path in item.items():
                    items.append((str(target_path), str(import_path)))
            else:
                raise TypeError(f"unsupported import item: {item}")
        return items

    if isinstance(value, (dict, DictConfig)):
        return [(str(target_path), str(import_path)) for target_path, import_path in value.items()]

    raise TypeError(f"imports/includes/include must be str, list, or dict, got {type(value)}")


def _get_import_items(cfg: DictConfig):
    items = []
    for key in _IMPORT_KEYS:
        if key in cfg:
            items.extend(_normalize_import_items(cfg.get(key)))
    return items


def _remove_import_keys(cfg: DictConfig) -> DictConfig:
    cfg = cfg.copy()
    for key in _IMPORT_KEYS:
        if key in cfg:
            del cfg[key]
    return cfg

def load_config(config_path: str, root_dir: str = None):
    config_path = Path(config_path).expanduser().resolve()

    if root_dir is None:
        root_dir = Path.cwd()
    else:
        root_dir = Path(root_dir).expanduser().resolve()

    cfg = _load_config_recursive(
        config_path,
        loading_stack=set(),
        root_dir=root_dir,
    )
    cfg = _resolve_extends(cfg)
    return cfg

def _load_config_recursive(config_path: Path, loading_stack: Set[Path], root_dir: Path):
    if config_path in loading_stack:
        cycle = " -> ".join(str(p) for p in list(loading_stack) + [config_path])
        raise RuntimeError(f"circular config import detected: {cycle}")

    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")

    loading_stack.add(config_path)

    cfg = OmegaConf.load(config_path)
    import_items = _get_import_items(cfg)

    merged = OmegaConf.create({})

    for target_path, import_path in import_items:
        import_path = Path(import_path).expanduser()

        if not import_path.is_absolute():
            import_path = root_dir / import_path

        import_cfg = _load_config_recursive(
            import_path.resolve(),
            loading_stack,
            root_dir=root_dir,
        )

        if target_path is None:
            merged = OmegaConf.merge(merged, import_cfg)
        else:
            OmegaConf.update(merged, target_path, import_cfg, merge=False)

    cfg = _remove_import_keys(cfg)
    merged = OmegaConf.merge(merged, cfg)

    loading_stack.remove(config_path)
    return merged

def _get_extends_value(node: DictConfig):
    for key in _EXTENDS_KEYS:
        if key in node:
            return key, node.get(key)
    return None, None


def _remove_extends_key(node: DictConfig) -> DictConfig:
    node = node.copy()
    for key in _EXTENDS_KEYS:
        if key in node:
            del node[key]
    return node


def _resolve_extends(cfg: DictConfig) -> DictConfig:
    return _resolve_node(cfg, cfg, path=(), resolving_stack=[])


def _resolve_node(node, root: DictConfig, path: Tuple[str, ...], resolving_stack: List[str]):
    if isinstance(node, DictConfig):
        path_str = ".".join(path)

        extends_key, extends_value = _get_extends_value(node)
        if extends_key is not None:
            if path_str in resolving_stack:
                cycle = " -> ".join(resolving_stack + [path_str])
                raise RuntimeError(f"circular config extends detected: {cycle}")

            merged_base = OmegaConf.create({})

            for base_ref in _to_list(extends_value):
                base_node = OmegaConf.select(root, base_ref)
                if base_node is None:
                    raise KeyError(f"extends target not found: {base_ref}")

                base_resolved = _resolve_node(
                    base_node,
                    root,
                    path=tuple(base_ref.split(".")),
                    resolving_stack=resolving_stack + [path_str],
                )
                merged_base = OmegaConf.merge(merged_base, base_resolved)

            override = _remove_extends_key(node)
            override = _resolve_node(
                override,
                root,
                path=path,
                resolving_stack=resolving_stack + [path_str],
            )

            return OmegaConf.merge(merged_base, override)

        resolved = OmegaConf.create({})
        for key, value in node.items():
            resolved[key] = _resolve_node(
                value,
                root,
                path=path + (str(key),),
                resolving_stack=resolving_stack,
            )
        return resolved

    if isinstance(node, ListConfig):
        return OmegaConf.create([
            _resolve_node(
                value,
                root,
                path=path + (str(i),),
                resolving_stack=resolving_stack,
            )
            for i, value in enumerate(node)
        ])

    return node