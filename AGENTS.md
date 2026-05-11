# context_compressor 模块指南

> 四层自适应上下文压缩引擎的内部架构与扩展指南。

## 架构概览

用户消息流 -> AdaptiveRouter._analyze_messages() -> AdaptiveRouter.route() -> 四层策略

- L1: SlidingWindowStrategy (token 超阈值)
- L2: IterativeSummaryStrategy (轮次超 20)
- L3: SemanticRetrievalStrategy (Mem0 可用)
- L4: CodeAwareCompressor / TokenCompressionStrategy

## 子包职责

- strategies/: 压缩策略层（router, sliding_window, iterative_summary, semantic_retrieval, token_compression）
- code_aware/: 代码感知压缩（python_compressor, python_detector）
- persistence/: 持久化层（mem0_adapter, session_store, session_state）
- utils/: 工具层（config, logger, messages, safety, token_counter）

## 如何添加新策略

1. 实现 StrategyProtocol
2. 在 AdaptiveRouter 中注册
3. 添加测试用例

## 编码规范

- Python 3.11+ 语法（X | None 替代 Optional[X]）
- 所有函数参数和返回值必须显式标注类型
- 行长度 ≤ 100 字符
- Docstring 使用 Google 风格（中文描述）
- 使用 ruff 进行代码格式化和 lint
- 使用 pyright strict 模式进行类型检查
