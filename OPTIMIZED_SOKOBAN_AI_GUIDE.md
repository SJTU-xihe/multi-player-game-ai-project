# 优化后的推箱子AI使用指南

## 概述

本项目现在包含多个优化的推箱子AI实现，包括基于LLM的智能AI和改进的搜索算法AI。

## AI类型介绍

### 1. LLMBot - 大语言模型AI
```python
from agents.ai_bots.llm_bot import LLMBot

# 创建LLM AI（使用本地模拟）
llm_ai = LLMBot(
    name="LLM推箱子AI",
    player_id=1,
    use_local_simulation=True,  # 使用本地模拟推理
    reasoning_depth=3           # 推理深度
)
```

**特点：**
- 基于语言模型的智能推理
- 能够理解游戏状态并做出策略性决策
- 支持本地模拟和真实LLM API调用
- 适合复杂场景的策略分析

### 2. AdvancedSokobanAI - 高级混合策略AI
```python
from agents.ai_bots.llm_bot import AdvancedSokobanAI

# 创建混合策略AI
hybrid_ai = AdvancedSokobanAI(
    name="混合策略AI",
    player_id=1,
    strategy='hybrid',          # 'llm', 'search', 'hybrid'
    search_depth=3
)
```

**特点：**
- 结合LLM推理和传统搜索算法
- 根据场景复杂度自动选择策略
- 复杂情况用LLM，简单情况用搜索
- 平衡性能和智能度

### 3. 优化的SokobanAI - 改进搜索算法
```python
from agents.ai_bots.sokoban_ai import SokobanAI

# 创建优化的搜索AI
optimized_ai = SokobanAI(
    name="优化搜索AI",
    player_id=1,
    max_search_time=3.0,        # 最大搜索时间
    max_depth=30,               # 最大搜索深度
    use_dynamic_depth=True,     # 动态深度调整
    cache_size=10000,           # 状态缓存大小
    use_heuristic=True          # 使用启发式函数
)
```

**特点：**
- 改进的A*搜索算法
- 动态深度调整
- 状态缓存和死锁检测
- 快速动作检查优化

### 4. SimpleSokobanAI - 简单贪心策略
```python
from agents.ai_bots.sokoban_ai import SimpleSokobanAI

# 创建简单AI
simple_ai = SimpleSokobanAI(
    name="简单贪心AI",
    player_id=1
)
```

**特点：**
- 简单快速的贪心策略
- 低计算开销
- 适合实时对战

## 使用示例

### 基本使用
```python
from agents.ai_bots.llm_bot import LLMBot, AdvancedSokobanAI
from agents.ai_bots.sokoban_ai import SokobanAI

# 创建不同类型的AI
ai_players = [
    LLMBot("LLM-AI", player_id=1, use_local_simulation=True),
    AdvancedSokobanAI("混合AI", player_id=1, strategy='hybrid'),
    SokobanAI("搜索AI", player_id=1, max_search_time=2.0)
]

# 在游戏中使用
for ai in ai_players:
    action = ai.get_action(observation, env)
    print(f"{ai.name} 选择动作: {action}")
```

### 性能调优
```python
# 高性能配置（适合复杂关卡）
high_performance_ai = SokobanAI(
    name="高性能AI",
    player_id=1,
    max_search_time=5.0,        # 更长搜索时间
    max_depth=50,               # 更深搜索
    use_dynamic_depth=True,
    cache_size=20000,           # 更大缓存
    use_heuristic=True
)

# 快速响应配置（适合实时对战）
fast_response_ai = SokobanAI(
    name="快速响应AI",
    player_id=1,
    max_search_time=1.0,        # 更短搜索时间
    max_depth=15,               # 较浅搜索
    use_dynamic_depth=True,
    cache_size=5000,
    use_heuristic=True
)
```

### LLM API配置
如果要使用真实的LLM API：
```python
# 使用OpenAI API
llm_ai = LLMBot(
    name="GPT-AI",
    player_id=1,
    model_name='gpt-3.5-turbo',
    api_key='your-openai-api-key',
    use_local_simulation=False,
    temperature=0.7
)

# 使用本地Ollama模型
local_llm_ai = LLMBot(
    name="本地LLM-AI",
    player_id=1,
    model_name='llama2',
    use_local_simulation=False,  # 设置为False并配置本地API
    api_key=None
)
```

## 性能对比

| AI类型 | 响应速度 | 智能度 | 资源消耗 | 适用场景 |
|--------|----------|--------|----------|----------|
| SimpleSokobanAI | 极快 | 低 | 极低 | 简单关卡，实时对战 |
| SokobanAI | 快 | 中高 | 中 | 一般关卡，平衡性能 |
| LLMBot | 中 | 高 | 中高 | 复杂关卡，策略分析 |
| AdvancedSokobanAI | 中 | 高 | 中 | 全场景，自适应 |

## 测试和验证

运行测试脚本来验证AI性能：
```bash
python test_optimized_sokoban_ai.py
```

测试内容包括：
- AI性能对比
- LLM推理能力测试
- 高级功能验证
- 不同场景适应性

## 配置建议

### 竞赛环境
- 使用 `AdvancedSokobanAI` 的混合策略
- 设置适中的搜索时间（2-3秒）
- 启用缓存优化

### 学习研究
- 使用 `LLMBot` 分析游戏策略
- 启用详细的推理输出
- 尝试不同的模型配置

### 实时游戏
- 使用 `SimpleSokobanAI` 或快速配置的 `SokobanAI`
- 限制搜索时间在1秒以内
- 优化缓存大小

## 扩展开发

### 自定义LLM推理
```python
class CustomLLMBot(LLMBot):
    def _sokoban_reasoning(self, observation, env):
        # 实现自定义推理逻辑
        return super()._sokoban_reasoning(observation, env)
```

### 自定义搜索策略
```python
class CustomSokobanAI(SokobanAI):
    def _evaluate_state(self, state):
        # 实现自定义状态评估
        return super()._evaluate_state(state)
```

## 注意事项

1. **内存管理**: 长时间运行时注意清理缓存
2. **API限制**: 使用真实LLM API时注意调用频率限制
3. **性能监控**: 监控AI响应时间，避免超时
4. **错误处理**: 实现robust的错误处理机制

## 故障排除

### 常见问题

1. **AI响应慢**
   - 减少搜索时间或深度
   - 清理缓存
   - 使用简单策略AI

2. **内存占用高**
   - 减少缓存大小
   - 定期清理缓存
   - 使用轻量级AI

3. **LLM调用失败**
   - 检查API密钥
   - 切换到本地模拟模式
   - 验证网络连接

4. **搜索效果差**
   - 调整启发式函数权重
   - 增加搜索深度
   - 优化状态评估函数

更多详细信息请参考源代码注释和示例。
