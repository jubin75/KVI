# 脚本说明：训练 Projector / Gate

目的：把 domain embedding 映射到目标模型 attention head_dim 空间，并学习/配置 γ（gate）。

## 输入
- `--encoder_ckpt`：domain_encoder
- `--target_model`：HF 模型名或路径（用于读取 head_dim/num_heads 等元信息）
- `--train_pairs`：训练样本（可用弱监督或合成 QA 派生）

## 关键参数（建议）
- `--projector`：linear | mlp | lora
- `--share_kv`：是否共享 K/V 投影（实现自行决定）
- `--gamma_mode`：constant | learned_layer | learned_head | learned_token
- `--gamma_clamp_max`：默认 0.10

## 输出
- projector 权重
- gate 参数（若 learned）
- 评测：注入后 QA 指标与稳定性回归项


