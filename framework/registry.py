OPS = {}


def register_op(name):
    def deco(cls_or_fn):
        OPS[name] = cls_or_fn
        return cls_or_fn
    return deco


def get_op(name):
    if name not in OPS:
        raise KeyError(f"unknown op type {name!r}. Registered ops: {sorted(OPS)}")
    return OPS[name]
