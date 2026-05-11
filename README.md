# Context Compressor

自动上下文压缩库，面向 LLM 代码 agent。

## 项目概述

`context-compressor` 是一个 Python 库，作为 LangGraph 中间件节点自动压缩 LLM agent 对话上下文，防止上下文遗忘。

## 核心能力

四层自适应压缩管道，从滑动窗口到代码感知截断逐级降维：

- **L1**: SlidingWindowStrategy（保留最近 N 条 + 粘滞笔记）
- **L2**: IterativeSummaryStrategy（LLM 增量摘要，锚定四字段）
- **L3**: SemanticRetrievalStrategy（Mem0 跨会话事实检索）
- **L4**: CodeAwareCompressor / TokenCompressionStrategy

## 技术栈

- Python >=3.11, <3.15
- LangGraph + LangChain
- Mem0（向量记忆）
- tiktoken（Token 计数）
- OpenAI（LLM 服务）

## 安装

```bash
pip install -e ".[dev]"
```

## 开发

```bash
make test   # 运行测试
make lint   # 代码检查
make build  # 构建分发包
```

## 许可证

MIT
