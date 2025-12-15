# 脚本说明：运行评测（AB/回归）

目的：执行评测协议（见 `docs/40_evaluation_protocol.md`），输出效果/延迟/稳定性/可解释性报告。

## 输入
- `--eval_set`：评测集路径
- `--config`：配置文件（如 `configs/example.yaml`）

## 关键参数（建议）
- `--ab_groups`：baseline（inject.off） vs injection（inject.on）
- `--k_values`：8,16,32
- `--export_debug`：是否输出引用与 γ 统计

## 输出
- 报告：Markdown/JSON
- 样例分析：命中 chunk 与引用正确性示例


