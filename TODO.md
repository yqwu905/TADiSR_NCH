# TADiSR 复现实现计划

基于 NCH MMDiT 模型 + f16c64 VAE，在 futuretrainer 框架上复现 TADiSR（Text-Aware Real-World Image Super-Resolution, NeurIPS 2025）。

## 核心技术决策

| 决策项 | 结论 |
|---|---|
| 目标方法 | TADiSR (Text-Aware, NeurIPS2025, arXiv:2506.04641) |
| 任务场景 | 文本图像 SR（保护文字笔画保真） |
| Prompt | 固定 `"A high-quality photo with clear text"` |
| TACA token 位置 | 方案 A：离线预分析 pangu 256 序列中 "text" 的 index |
| 文本编码 | 离线 EmbeddingDB（查 SQLite 缓存，pangu 256×1536） |
| 数据 | 先用随机数据跑通 pipeline，FTSR 数据本地接入 |
| DiT 微调 | 全参数微调 |
| TACA attention | 替换 SDPA 为手动 attention 暴露 weights |
| JSD 分割解码器 | 新建独立组件（图像分支复用 Decoder+VAE权重，分割分支随机初始化，CDIB 连接） |
| DiT 输入构造 | `concat[lq_latent(64), mask(128), lq_latent(64)]` = 256ch → patchify 2×2 → 1024ch |

---

## 阶段 0：基础设施与依赖（已完成）

### 0.1 框架迁移 ✅
- [x] Clone fictional-fortnight 框架
- [x] 迁移 framework/configs/test_assets/tests 到项目根目录
- [x] 配置 pyproject.toml（torch 2.9.0 + diffusers 依赖）

### 0.2 NCH MMDiT 模型整合 ✅
- [x] `models/dit/nch/ldm/transformer_nch_v3_split.py` — NCHTransformer2DModel
- [x] `models/dit/nch/ldm/attention_processor_v3_split.py` — Attention + NCHAttnProcessor2_0/SparseAttnProcessor
- [x] `models/dit/nch/ldm/attention.py` — FeedForward
- [x] `models/dit/nch/ldm/normalization.py` — AdaLayerNormZero/AdaLayerNormContinuous/RMSNorm
- [x] `models/dit/nch/ldm/embeddings.py` — CombinedTimestepEmbeddings/QwenEmbedRope
- [x] `models/dit/nch/ldm/bmm.py` — SparseProcessAttnAigc 稀疏注意力
- [x] `models/dit/nch/ldm/npu_utils.py` — fast_interleave
- [x] `models/dit/nch/ldm/dump_config.py` — 调试 dump 配置
- [x] 修复 3 处因目录移动的 import 路径
- [x] 修复 unrearrange 笔误（`patch_size, patch_size` → `patch_size * patch_size`）

### 0.3 VAE 整合 ✅
- [x] `models/vae/npu/mj64_vae.py` — Encoder/Decoder/AutoencoderKL 原始实现
- [x] `models/vae/npu/f16c64.py` — VaeEncoder/VaeDecoder 包装类（latent scaling + FSDP wrap）

### 0.4 文本编码组件 ✅
- [x] `models/text_encoder/offline_embedding.py` — EmbeddingDB（清理后，无 NPU 依赖）
- [x] `models/text_encoder/nch/utils_nch.py` — TrainableVector_multitask（补 import os/torch）

### 0.5 数据集组件 ✅
- [x] `data/tadisr_dataset.py` — TADiSRDataset（synthetic + real 模式）+ TADiSRCollateFn
- [x] 框架 build_dataloader 支持 collate_fn 配置

### 0.6 Pangu token 位置分析脚本 ✅
- [x] `scripts/analyze_pangu_text_token.py` — 确定固定 prompt 中 "text" 在 256 序列的 index

---

## 阶段 1：SR 基线管线（已完成）

### 1.1 nch_mmdit_sr op ✅
- [x] `framework/ops/diffusion.py` 新增 `nch_mmdit_sr` op
- [x] 适配新版 transformer forward 签名（encoder_hidden_states + Transformer2DModelOutput）
- [x] LQ-as-start 输入拼接 [lq, mask, lq]
- [x] device 转移处理（EmbeddingDB CPU → latent device）
- [x] flow-matching 单步去噪（x0 = x_start - dt * v）

### 1.2 临时 SR Loss ✅
- [x] `models/loss/sr_loss.py` — SRLoss（L2 + 可选 LPIPS，graceful fallback）

### 1.3 SR 训练配置 ✅
- [x] `configs/tadisr_sr_baseline.yaml` — 完整单 phase 配置
- [x] 组件：vae_encoder(frozen) + vae_decoder(frozen) + offline_embedding(frozen) + dit(train)
- [x] Phase: vae_encode → text_encode → dit_sr → vae_decode → sr_loss

### 1.4 验证 ✅
- [x] compileall 全部通过
- [x] 29 个 unittest 全部通过
- [x] 配置加载通过
- [x] 端到端 CPU smoke 训练 2 步成功（default processor，loss 正常下降）
- [ ] sparse processor 需 ≥1024 分辨率（Colab GPU 验证）

---

## 阶段 2：TACA 文本感知（已实现，Colab GPU 验证完成）

### 2.1 Pangu token 位置分析 ✅
- [x] 在 pangu 环境运行 `scripts/analyze_pangu_text_token.py`
- [x] 结果：prompt "A high-quality photo with clear text" 在 256 序列中 index=8
- [x] 256 序列结构：[1,1] 特殊 token + [0-6] 7 个 prompt token + [8] "▁text" + [9-254] padding + [255] eos
- [x] TACA text_token_indices = [8]（配置中设置）

### 2.2 Attention processor 改造 ✅（chunked logsumexp 重构）
- [x] `NCHAttnProcessor2_0`：新增 `taca_config` 旁路模式（chunked logsumexp，image→text 方向）
- [x] 主路径始终走 fused SDPA，旁路只算 text attention slice，不实例化全矩阵
- [x] 旁路支持 query chunking + gradient checkpointing（`query_chunk_size`, `use_checkpoint`）
- [x] `SparseAttnProcessor`：保留旧 `store_attn_weights` 全矩阵路径（后续再适配）
- [x] 验证 chunked logsumexp 与全矩阵数值一致性（fp32 max_diff=2.56e-09，atol=1e-5 PASS）
- [x] 验证主路径 SDPA 输出在 extract 开关下不变（atol=1e-6）

### 2.3 TACA 投影头 ✅
- [x] `models/dit/nch/ldm/taca.py` — `TACAProjection` 接受预提取 slice `[B,H,S_img,n_text]`
- [x] 拼接多层 slice → 线性投影 → `pred.a_tex` `[B,S_img,out_dim]`
- [x] 零初始化，保证不扰动预训练路径（weight/bias 全零，初始 a_tex=0）

### 2.4 TACA 提取集成 ✅
- [x] `NCHTransformer2DModel` forward：设置 `taca_config` → 收集 `taca_slice` → 投影 a_tex
- [x] 扩展 `nch_mmdit_sr` op 支持 `extract_a_tex`，把 a_tex 写入 ctx（`pred.a_tex`）

### 2.5 验证 ✅
- [x] `compileall` 全部通过（framework/test_assets/tests/models，CPU + Colab GPU）
- [x] 39 个 unittest 全部通过（含 9 个 TACA 测试：投影头 shape/零初始化/梯度、
      legacy 手动 vs SDPA、chunked vs 全矩阵数值一致性、主路径不变、slice shape/概率范围、
      DiT extract a_tex、op 胶水）
- [x] 不破坏 SR 基线（基线路径仍走 SDPA，返回 Transformer2DModelOutput）
- [x] Colab GPU 数值一致性（A100-40GB, fp32）：chunked logsumexp vs 全矩阵 max_diff=2.56e-09

### 2.6 Colab GPU 显存 benchmark ✅（A100-40GB, bf16, no_grad）

单层 attention 显存对比（旧全矩阵 vs 新 chunked logsumexp）：

| 分辨率 | S_img | 旧方法(MB) | 新方法(MB) | 节省 |
|---|---|---|---|---|
| 128 | 4096 | 345.5 | 198.5 | 2x |
| 256 | 16384 | 4585.9 | 752.2 | 6x |
| 384 | 36864 | 22390.8 | 1675.2 | 13x |
| 512 | 65536 | OOM | 2967.2 | ∞ |
| 1024 | 262144 | — | 11827.2 | — |

37 层 DiT + TACA extract 端到端显存扫描（3 层 TACA, chunk=1024）：

| 分辨率 | S_img | 峰值(GB) | Fwd+TACA(GB) |
|---|---|---|---|
| 128 | 4096 | 15.13 | 0.29 |
| 256 | 16384 | 15.97 | 1.13 |
| 384 | 36864 | 17.46 | 2.62 |
| 512 | 65536 | 19.63 | 4.78 |
| 640 | 102400 | 22.28 | 7.43 |
| 768 | 147456 | 25.39 | 10.54 |

- [x] 旧方法 512×512 OOM（weights 矩阵 ~34GB）；新方法 768×768 仅 25.39GB
- [x] A100-40GB 推理上限：768×768（25.39GB，还有 ~15GB 余量）
- [x] 训练（带 backward）预计 512×512 可行（推理 19.63GB + 梯度 ~10-15GB）

---

## 阶段 3：JSD 分割解码器（已实现，CPU smoke 验证完成）

### 3.1 JSD 组件 ✅
- [x] 新增 `models/vae/npu/jsd.py` — `JointSegDecoder` + `CDIB`
- [x] 图像分支：复用 `mj64_vae.Decoder` 结构，支持 `image_checkpoint` 加载 VAE 权重
- [x] 分割分支：随机初始化的对称 `Decoder`（`seg_channels=1`，sigmoid 输出 mask）
- [x] CDIB（Cross-Decoder Interaction Block）：双分支交互
  - ResBlock → 1×1 Conv(2C) → Split → Hadamard × Sigmoid(交叉半) → GroupNorm + SiLU + 1×1 Conv → 零初始化残差（`scale_img`/`scale_seg` init 0，训练初期为 identity）
- [x] CDIB 按 up-sampling level 插入，每个 level 的 channels 按 `ch*ch_mult[level]` 自动匹配（默认 `[4,3,2,1]`）
- [x] a_tex 序列→空间 reshape：`[B,S_img,C]` → 推断 patch 网格 → `interpolate` 对齐 latent 空间
- [x] image branch 输出应用 VAE scaling（`1/SCALING_FACTOR * out + SHIFTING_FACTOR`），与 `VaeDecoder` 一致

### 3.2 jsd_decode op ✅
- [x] 新增 `jsd_decode` op（`framework/ops/diffusion.py`）：输入 latent + a_tex → `pred.rgb` + `pred.seg`
- [x] 在 `configs/tadisr_jsd.yaml` 添加 JSD 组件，DiT 开启 `taca_cfg`（`extract_a_tex=true`）

### 3.3 验证 ✅
- [x] `compileall` 全部通过（framework/test_assets/tests/models）
- [x] 51 个 unittest 全部通过（含 12 个 JSD 测试：CDIB 零初始化 identity/shape/梯度、
      JSD 输出 shape/seg 范围/a_tex_channels 灵活/梯度流/错误 seq len/默认 cdib_levels/
      无 CDIB 时 image branch == VAE decode、op 胶水/错误处理）
- [x] 端到端 CPU smoke 训练 2 步成功（4 层 DiT + TACA extract + JSD，loss 0.44→0.13 正常下降）
- [x] JSD 参数量 170.765M（两个 VAE Decoder + 4 CDIB），DiT 4 层 367.078M
- [ ] Colab GPU smoke（JSD 在 512/1024 分辨率下的显存与质量验证）

---

## 阶段 4：完整 Loss + 训练（待实现）

### 4.1 TADiSRLoss ⬜
- [ ] 新增 `models/loss/tadisr_loss.py`
- [ ] L2 + 5.0·LPIPS + 10.0·modified_focal（Sobel 边缘）
- [ ] modified focal: `‖[1 - ŝ∘s - (1-ŝ)∘(1-s)]^γ ∘ (∇x̂ - ∇x)²‖₁`

### 4.2 SegLoss ⬜
- [ ] 新增 `models/loss/seg_loss.py`
- [ ] L2 + 10.0·Focal + 1.0·Dice

### 4.3 完整训练配置 ⬜
- [ ] `configs/tadisr_full.yaml` — 多 loss + DiT + SegDecoder 联合训练
- [ ] Phase: vae_encode → text_encode → dit_sr → vae_decode + seg_decode → sr_loss + seg_loss
- [ ] lr=5e-5, AdamW, DiT + SegDecoder 联合优化

### 4.4 验证 ⬜
- [ ] Colab GPU 完整 smoke 训练
- [ ] 所有 loss 项正常计算
- [ ] checkpoint 保存/加载

---

## 阶段 5：Colab GPU 调试（已完成）

### 5.1 环境搭建 ✅
- [x] 连接 Colab GPU session (L4 23.5GB, torch 2.11.0+cu128)
- [x] 推送到 GitHub (https://github.com/yqwu905/TADiSR_NCH)
- [x] Colab 克隆仓库 + 安装依赖
- [x] 语法检查 + smoke 测试通过

### 5.2 SR 基线 GPU 验证 ✅
- [x] DiT sparse processor + 1024 分辨率 forward 通过（输出 [1,64,64,64]，1.49GB 显存）
- [x] 完整管线 4 层 DiT (512 res, bf16, default processor) 端到端训练 2 步通过
- [x] 37 层 DiT forward+backward 通过（VAE decode 阶段 OOM，L4 显存限制，实际用 A100 或开 gradient_checkpointing）

### 5.3 已知限制
- sparse processor 要求 ≥1024 分辨率（block_lenth=64 导致 topk 越界）
- L4 24GB 无法跑完整 37 层 + 1024 分辨率（需 A100 或 gradient_checkpointing）
- 无 sqlite 缓存时 EmbeddingDB 返回零张量（smoke 可用，实际训练需提供）
- TACA 手动 attention 在 ≥512 分辨率时 OOM（weights 矩阵 ~34GB @512²）；
  A100-40GB 下 128×128 bf16 可跑（峰值 21.8GB），256² 需减少 TACA 层数或开
  gradient_checkpointing。后续可优化为 chunked/行采样 attention 降低显存。

---

## 阻塞项追踪

| 阻塞项 | 状态 | 说明 |
|---|---|---|
| Pangu 分析环境 | ✅ 已完成 | index=8，阶段 2 TACA token 位置已确定 |
| SQLite 缓存文件 | ⬜ 待用户提供路径 | 阻塞实际训练（smoke 用零张量可跑） |
| FTSR 数据集 | ⬜ 待用户本地接入 | 阻塞实际训练（smoke 用随机数据可跑） |
| DiT/VAE checkpoint | 可选 | 实际训练时填路径，smoke 用随机初始化 |

## 非阻塞项

- DiT/VAE checkpoint — 随机初始化可验证管线
- GPU — CPU smoke 验证 shape/loss/数据流
- PanguTokenizer 完整代码 — 训练不需要（EmbeddingDB 直接查 SQLite）
