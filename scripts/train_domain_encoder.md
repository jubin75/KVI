# 脚本说明：训练 DomainEncoder（检索 encoder）

目的：训练/蒸馏一个领域检索 encoder，用于 chunk 检索（Recall@k 为主指标）。

## 输入
- `--train_data`：检索对数据（query, positive, negatives）
- `--valid_data`：验证集

## 关键参数（建议）
- `--model`：bert/scibert/mini-lm/...（实现自行选择）
- `--batch_size`：按显存与序列长度调参
- `--negatives`：in-batch | mined-hard
- `--max_length`：与 chunk token 上限一致或略小

## 输出
- encoder 权重
- 指标报告：Recall@k（k=8/16/32）、MRR@k（可选）


