# AGENTS.md

本项目是一个模块化 PyTorch 训练框架。当前仓库只保留了框架代码和部分示例配置，具体的 `models/`、`network/`、`data/` 等业务实现已经移除；因此文档中的训练命令需要在补回对应模块、数据和 checkpoint 后才能完整运行。

## 项目结构

- `framework/train.py`: 命令行入口。读取 YAML 配置，合并 dotlist overrides，创建 `Trainer` 并启动训练。
- `framework/config.py`: 配置加载层。支持文件级 `imports`、`includes`、`include`，以及节点级 `extends`、`_base_`。
- `framework/engine.py`: 训练主循环。负责分布式初始化、DataLoader、组件构建、日志、checkpoint、TensorBoard 图像写入。
- `framework/components.py`: 组件生命周期管理。负责 `target + params` 实例化、checkpoint 加载、冻结/解冻、DDP 包装、参数统计。
- `framework/phase_runner.py`: 训练阶段执行器。按 phase 顺序执行 op、loss、backward、grad clip、optimizer step、scheduler step。
- `framework/ops/`: 内置 op 注册点。`common.py` 提供通用调用/写 context 操作，`diffusion.py` 提供扩散/flow matching 辅助操作。
- `framework/losses.py`: loss 包装器。用配置把 context 中的数据映射到 loss 函数参数，并返回加权 loss 与 metrics。
- `framework/optim.py`: optimizer 和 scheduler 构建。
- `framework/resolver.py`: context 输入解析器。支持 `ctx`、`const`、`ones_like`、`zeros_like`、`randn_like`、`detach`、`cast`、`getattr` 等输入 spec。
- `framework/distributed.py`: CPU/CUDA/NPU 设备推断与 DDP 初始化。
- `configs/`: 示例配置。当前部分引用了已删除的业务模块或未保留的外部配置，只能作为配置结构参考。

## 框架模型

训练由四类对象组成：

1. `components`: 可训练或不可训练的模块，例如 VAE、DiT、text encoder、embedding DB。每个组件通过 `target` 全路径导入并用 `params` 初始化。
2. `train_program.phases`: 一个 step 内可以有多个 phase。每个 phase 定义本阶段可训练组件、冻结组件、模式、op 列表、loss 列表和优化器动作。
3. `ops`: 数据流编排单元。op 从 `TrainContext` 读取输入，调用组件或函数，再把输出写回 context。
4. `losses`: 从 context 读取输入，计算 loss，并把 metrics 交给主循环记录。

`TrainContext` 使用点分路径保存中间状态，例如 `batch.gt`、`pred.rgb`、`noise.t`。新增配置时应保持 context key 命名稳定、可读、按领域分组。

## 配置规范

组件配置使用统一形态：

```yaml
components:
  my_component:
    target: package.module.ClassName
    params:
      arg1: value
    checkpoint: /optional/path.pt
    strict: true
    train:
      strategy: full
    mode: train
    save: full
    gradient_checkpointing: true
```

约定：

- `target` 必须是完整 Python import path，且构造函数参数只能来自 `params` 或框架注入参数。
- `train.strategy` 支持 `full`、`frozen`、`keep`。LoRA 等参数高效微调的挂载交给组件自身初始化实现。
- `mode` 支持 `train`、`eval`，只控制初始模式；phase 内的 `modes` 可覆盖。
- `save` 支持 `full`、`none`，LoRA 模块可使用 `lora_only`。
- checkpoint 可以是裸 `state_dict`，也可以是包含 `state_dict` key 的 dict。
- `gradient_checkpointing` 可按组件独立开启，用于节省显存。支持 `true` 简写或 `{ enabled, method, method_kwargs }` 完整形式。默认自动探测模块的 `gradient_checkpointing_enable` / `enable_gradient_checkpointing` 方法并无参调用（走库默认值）；需要自定义时用 `method` 指定方法名、`method_kwargs` 透传参数。在 DDP/FSDP2 包装前启用，FSDP2 下建议保持 `use_reentrant=false`。未实现该方法会报错。

配置复用：

- 文件级导入：

```yaml
imports:
  components.dit: configs/network_config/ldm_nch_v3.yaml
  data.train.dataset: configs/data_config/my_dataset.yaml
```

- 节点级继承：

```yaml
my_node:
  extends: base_node
  params:
    override_arg: value
```

注意：相对路径按运行命令所在目录解析。默认从仓库根目录运行命令。

## 编码规范

- Python 版本按 `pyproject.toml` 要求使用 `>=3.10`。
- 新增文件默认使用 ASCII；只有已有文件或业务语义需要时才使用非 ASCII。
- 模块、函数、变量使用 `snake_case`，类使用 `PascalCase`，常量使用 `UPPER_SNAKE_CASE`。
- 新增组件、数据集、loss 应优先实现为普通 Python 类或 `torch.nn.Module`，并通过 `target + params` 暴露给配置。
- 构造函数不要读取全局配置；所有可变行为都应放进 `params` 或 phase 配置。
- 训练中间结果通过 `ctx.set("group.name", value)` 写入，避免扁平 key 和临时缩写。
- op 应保持小而明确：解析输入、调用一个动作、写出结果。复杂业务逻辑放到业务模块或新 op 中，不要塞进 YAML 的过长调用链。
- loss 返回值可以是 Tensor，或包含 `loss` key 的 dict；额外 key 会被记录为 metrics。
- 分布式训练中只在 main process 写文件、打印大段日志或保存 checkpoint。
- 新增 DataLoader 配置时，优先显式设置 `batch_size`、`shuffle`、`num_workers`、`drop_last`。DDP 下 sampler 会接管 shuffle。

## 新增扩展点

新增业务组件：

1. 在业务包中实现类，例如 `models.foo.FooModel`。
2. 构造参数全部写入 YAML 的 `params`。
3. 如果需要加载权重，使用组件级 `checkpoint` 和 `strict`。
4. 在 `optimizers.*.params.include` 中引用组件名或 `component.trainable`。

新增 op：

```python
from framework.registry import register_op
from framework.resolver import resolve_kwargs

@register_op("my_op")
class MyOp:
    def __init__(self, cfg):
        self.cfg = dict(cfg)

    def __call__(self, ctx, components):
        kwargs = resolve_kwargs(self.cfg.get("inputs", {}), ctx)
        result = ...
        ctx.set(self.cfg["output"], result)
```

新增 loss：

1. 实现 `torch.nn.Module` 或可调用对象。
2. 参数来自 YAML `params`。
3. 前向参数由 `losses.<name>.inputs` 映射。
4. 返回 Tensor 或 `{"loss": tensor, "metric_name": value}`。

## 工具链建议

当前 `pyproject.toml` 只声明运行依赖，尚未配置代码质量工具。建议后续补齐：

- `ruff`: 格式化与 lint，替代大部分手写风格检查。
- `pyright` 或 `basedpyright`: 静态类型检查，至少覆盖 `framework/`。
- `pytest`: 单元测试与 smoke test。
- `tensorboard`: 已在依赖中声明，用于训练日志查看。
- `uv`: 项目已有 `uv.lock`，优先用 `uv sync` 和 `uv run` 管理环境。

建议的最小质量门禁：

```bash
uv run ruff format framework
uv run ruff check framework
uv run pyright framework
uv run pytest
PYTHONPYCACHEPREFIX=/tmp/fictional-fortnight-pycache python3 -m compileall -q framework test_assets tests
python3 -m unittest discover -s tests -v
```

每次修改代码或配置后，都必须运行测试。若环境中暂时没有安装 ruff、pyright、pytest，至少执行 `compileall` 和 `unittest` 两条命令做语法级与 smoke 验证。

## 安装与环境

创建或同步环境：

```bash
uv sync
```

如果需要 NPU 依赖：

```bash
uv sync --group npu
```

查看入口参数：

```bash
uv run python -m framework.train --help
```

如果 `uv` 环境暂不可用，也可以使用当前解释器做框架语法检查：

```bash
PYTHONPYCACHEPREFIX=/tmp/fictional-fortnight-pycache python3 -m compileall -q framework test_assets tests
python3 -m unittest discover -s tests -v
```

`PYTHONPYCACHEPREFIX` 是为了避免 macOS Python 把缓存写入受限的用户缓存目录。

## 运行命令

单进程 CPU 或单卡调试，覆盖配置里的 DDP：

```bash
uv run python -m framework.train \
  --config configs/f16c64_vae_dit_proj_out_align.yaml \
  runtime.distributed.strategy=none \
  runtime.device=cpu \
  train.max_steps=2 \
  train.log_every=1 \
  data.train.dataloader.num_workers=0
```

单机多卡 DDP：

```bash
uv run torchrun --nproc_per_node=2 -m framework.train \
  --config configs/f16c64_vae_dit_proj_out_align.yaml \
  runtime.distributed.strategy=ddp \
  train.max_steps=100
```

启用 profiling：

```bash
uv run python -m framework.train \
  --config configs/f16c64_vae_dit_proj_out_align.yaml \
  runtime.distributed.strategy=none \
  profiling.enabled=true \
  profiling.log_every=1 \
  train.max_steps=5
```

查看 TensorBoard：

```bash
uv run tensorboard --logdir output
```

注意：当前示例训练配置引用了已删除的业务实现，例如 `models.*`、`network.*`、`data.*`，并且部分 `imports` 指向未保留文件。补回这些模块和配置前，训练命令会在 import、配置加载或 checkpoint 路径处失败。

## 调试命令

验证框架 Python 语法：

```bash
PYTHONPYCACHEPREFIX=/tmp/fictional-fortnight-pycache python3 -m compileall -q framework test_assets tests
```

运行当前 smoke 测试：

```bash
python3 -m unittest discover -s tests -v
```

检查配置是否能完成 `imports` 和 `extends` 解析：

```bash
uv run python - <<'PY'
from framework.config import load_config
cfg = load_config("configs/f16c64_vae_dit_proj_out_align.yaml")
print(cfg)
PY
```

列出已注册 op：

```bash
uv run python - <<'PY'
import framework.ops
from framework.registry import OPS
print(sorted(OPS))
PY
```

检查组件参数选择是否为空，需要业务模块可导入：

```bash
uv run python - <<'PY'
from framework.config import load_config
from framework.components import ComponentManager
from framework.optim import build_optimizers

cfg = load_config("configs/f16c64_vae_dit_proj_out_align.yaml")
components = ComponentManager(cfg.components).build_all()
components.print_parameter_summary()
optimizers = build_optimizers(cfg.optimizers, components)
print(sorted(optimizers))
PY
```

定位 context 缺失 key：

- 优先看异常里的 `Available keys`。
- 检查前置 op 的 `outputs` 是否写到了 loss 或后续 op 期望的路径。
- 检查 `losses.*.inputs` 和 `ops.*.inputs` 是否使用了同一套 context key。

定位 DDP 问题：

- 未用 `torchrun` 启动时，不要设置 `runtime.distributed.strategy=ddp`。
- 单进程调试先覆盖为 `runtime.distributed.strategy=none`。
- 如果 phase 中有条件分支或未使用参数，尝试设置组件级或全局 DDP `find_unused_parameters=true`。

定位 DataLoader 问题：

- 先覆盖 `data.train.dataloader.num_workers=0`，把错误暴露到主进程。
- 再逐步恢复 `num_workers`、`persistent_workers`、`prefetch_factor`。

## 当前已知状态

- `README.md` 为空，`AGENTS.md` 是当前主要项目说明入口。
- `framework/` 通过了语法级编译检查：

```bash
PYTHONPYCACHEPREFIX=/tmp/fictional-fortnight-pycache python3 -m compileall -q framework
```

- 示例配置中 `configs/f16c64_vae_dit_proj_out_align.yaml` 和 `configs/f16c64_vae_x_embbder_align.yaml` 的部分 import 目标不存在于当前仓库。
- `framework/ops/common.py` 的 `save_image` op 当前引用了未在文件中定义或导入的 `save_image`、`_ensure_list`、`_basename_without_ext`。使用该 op 前需要补齐实现或导入。

## Repository Map

A full codemap is available at `codemap.md` in the project root.

Before working on any task, read `codemap.md` to understand:
- Project architecture and entry points
- Directory responsibilities and design patterns
- Data flow and integration points between modules

For deep work on a specific folder, also read that folder's `codemap.md`.
