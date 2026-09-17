# 宿主契约的 vendor 副本

`host_memory_provider.py` 是 **Hermes Agent** 的 `agent/memory_provider.py` 原样副本
（源头：`NousResearch/hermes-agent`，MIT）。

为什么要把别人的代码抄进来：**契约是一份外部事实，不是我们的记忆。**
测试需要一条能机械比对的基准；如果只把"宿主有哪些钩子"写在注释里，
下一次改代码的人没有任何东西可以比对，契约会以每年一个方法的速度悄悄漂掉。

刷新方式：

    python scripts/realtest/contract_diff.py --fetch

该命令会从上游重新拉取并覆盖此文件，然后与本项目的 provider 逐条比对。

## `host_tool_schema.py`

宿主 `agent/memory_manager.py` 的 `normalize_tool_schema` 原样副本。

它定义了**工具 schema 的正确形状**：宿主把它 `get_tool_schemas()` 的返回值
**原样**塞进 `{"type": "function", "function": schema}` 交给模型，而 OpenAI 规范读的是
`function.parameters`。

抄这一份的意义在于：形状错误的后果是**静默的**——工具照样能被调用，只是模型收不到
参数定义，表现为"参数传不进去"。真机实测时模型的原话就是这个。
`tests/test_host_contract.py` 用它断言每个工具都真有 `parameters`，
并且**必填参数在 `properties` 里可见**。

刷新方式同 `host_memory_provider.py`：`python scripts/realtest/contract_diff.py --fetch`。
