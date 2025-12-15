# 配置说明（configs）

本目录存放配置字段定义与示例配置文件（仅字段与注释说明，不包含实现代码）。

## 约定
- 所有配置应支持版本化与可复现（experiment_id、dataset_version、index_version 等）
- 线上推理需支持 AB：`inject.enabled`、`inject.strategy`、`inject.gamma`、`inject.layers` 等可动态切换（实现自行决定热更新方式）


