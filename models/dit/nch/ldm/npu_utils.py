def fast_interleave(tensor, repeat, dim):
    """
    通用型 NPU 友好维度重复函数 (替代 repeat_interleave)
    支持负维度
    """
    if repeat == 1:
        return tensor

    ndim = tensor.dim()

    # 处理负维度
    if dim < 0:
        dim += ndim

    old_shape = list(tensor.shape)

    # 1. 在目标维度后插入新维度
    tensor = tensor.unsqueeze(dim + 1)

    # 2. expand
    expand_shape = [-1] * tensor.dim()
    expand_shape[dim + 1] = repeat
    tensor = tensor.expand(*expand_shape)

    # 3. reshape
    new_shape = (
        old_shape[:dim]
        + [old_shape[dim] * repeat]
        + old_shape[dim + 1:]
    )

    return tensor.reshape(*new_shape)

def fast_interleave_2d(tensor, block_length, dims=(2, 3)):
    """
    在指定的多个维度上，同时进行 NPU 友好的交织扩展
    """
    for d in dims:
        tensor = fast_interleave(tensor, block_length, dim=d)
    return tensor

if __name__ == "__main__":
    import torch
    x = torch.tensor([[1, 2],
                  [3, 4]])

    print(fast_interleave_2d(x, 2, (-1,-2)))
