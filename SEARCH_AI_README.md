# 搜索算法AI使用指南

## 概述

搜索算法AI (`SearchAI`) 是一个通用的游戏智能体，实现了多种经典搜索算法，包括广度优先搜索 (BFS)、深度优先搜索 (DFS) 和 A* 搜索算法。该AI能够自动适应不同的游戏类型，为每种游戏提供相应的搜索策略。

## 支持的搜索算法

### 1. 广度优先搜索 (BFS)
- **算法特点**: 保证找到最短路径
- **适用场景**: 寻找最优解，内存充足的情况
- **时间复杂度**: O(b^d)
- **空间复杂度**: O(b^d)

### 2. 深度优先搜索 (DFS)
- **算法特点**: 搜索深度大，内存消耗少
- **适用场景**: 深度搜索，解空间较大的情况
- **时间复杂度**: O(b^m)
- **空间复杂度**: O(bm)

### 3. A* 搜索算法
- **算法特点**: 结合启发式函数，通常性能最优
- **适用场景**: 有明确目标的路径搜索
- **时间复杂度**: O(b^d)（在良好启发式下更优）
- **空间复杂度**: O(b^d)

## 支持的游戏类型

### 1. 贪吃蛇 (Snake)
- **搜索目标**: 找到从蛇头到食物的最优路径
- **避障处理**: 自动避开蛇身和对手
- **安全机制**: 如果无法找到路径，选择安全方向

### 2. 推箱子 (Sokoban)
- **搜索目标**: 找到推动箱子到目标位置的路径
- **策略**: 简化的启发式搜索
- **复杂度**: 适用于简单到中等复杂度的关卡

### 3. 五子棋 (Gomoku)
- **搜索目标**: 评估棋盘位置得分
- **策略**: 位置评估和连子检测
- **深度**: 可配置的搜索深度

## 使用方法

### 1. 命令行使用

```bash
# 使用默认BFS算法
python main.py --game snake --player1 search --player2 random

# 使用特定搜索算法
python main.py --game snake --player1 search_bfs --player2 random
python main.py --game snake --player1 search_dfs --player2 random
python main.py --game snake --player1 search_astar --player2 random

# 五子棋游戏
python main.py --game gomoku --player1 search_astar --player2 random

# 批量测试
python main.py --game snake --player1 search_bfs --player2 random --games 10 --no-render
```

### 2. 程序中直接使用

```python
from agents.ai_bots.search_ai import SearchAI

# 创建BFS搜索AI
bfs_ai = SearchAI(
    name="BFS-AI",
    player_id=1,
    search_algorithm="bfs",
    max_depth=15
)

# 创建A*搜索AI
astar_ai = SearchAI(
    name="A*-AI",
    player_id=1,
    search_algorithm="astar",
    max_depth=20
)

# 创建DFS搜索AI
dfs_ai = SearchAI(
    name="DFS-AI",
    player_id=1,
    search_algorithm="dfs",
    max_depth=25
)
```

### 3. GUI模式使用

在图形界面模式下，可以在AI选择菜单中找到搜索算法选项：
- **Search BFS** - 广度优先搜索
- **Search DFS** - 深度优先搜索 (仅贪吃蛇游戏)
- **Search A*** - A*搜索算法

> **注意**: 不同游戏显示的搜索AI选项可能不同。贪吃蛇游戏显示全部三种算法，五子棋和推箱子显示BFS和A*算法。

## 配置参数

### 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | str | "SearchAI" | AI名称 |
| `player_id` | int | 1 | 玩家ID |
| `search_algorithm` | str | "bfs" | 搜索算法类型 |
| `max_depth` | int | 10 | 最大搜索深度 |

### 搜索算法选项

- `"bfs"` - 广度优先搜索
- `"dfs"` - 深度优先搜索
- `"astar"` - A*搜索算法

## 性能调优

### 1. 调整搜索深度

```python
# 适合简单游戏的设置
shallow_ai = SearchAI(search_algorithm="bfs", max_depth=10)

# 适合复杂游戏的设置
deep_ai = SearchAI(search_algorithm="astar", max_depth=30)
```

### 2. 选择合适的算法

- **实时游戏**: 推荐使用 BFS 或 A*，深度设置为 10-15
- **回合制游戏**: 可以使用更大的搜索深度，15-30
- **复杂环境**: 推荐使用 A*，结合启发式函数

### 3. 内存优化

- DFS 算法内存消耗最少
- BFS 和 A* 内存消耗较大，注意深度设置
- 大地图建议降低 max_depth 参数

## 测试和验证

### 1. 基础功能测试

```bash
python test_search_ai_simple.py
```

### 2. 完整性能测试

```bash
python test_search_ai.py
```

### 3. 示例演示

```bash
python examples/search_ai_examples.py
```

## 性能对比

根据测试结果，不同算法的特点：

| 算法 | 贪吃蛇胜率 | 平均决策时间 | 内存使用 | 最优性 |
|------|------------|--------------|----------|--------|
| BFS | 中等 | 快 | 高 | 最优 |
| DFS | 较低 | 很快 | 低 | 非最优 |
| A* | 高 | 快 | 中等 | 接近最优 |

## 扩展和自定义

### 1. 添加新的启发式函数

```python
def custom_heuristic(self, pos1, pos2):
    # 自定义启发式函数
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
```

### 2. 针对特定游戏优化

可以通过继承 `SearchAI` 类并重写特定游戏的搜索方法来优化性能。

### 3. 添加新游戏支持

在 `_detect_game_type` 方法中添加新的游戏类型检测逻辑。

## 常见问题

### Q: 为什么搜索AI有时表现不如随机AI？
A: 这可能是因为搜索深度不够，或者游戏环境变化太快。建议增加 max_depth 或选择更适合的算法。

### Q: 如何提高搜索AI的性能？
A: 1) 选择合适的搜索算法；2) 调整搜索深度；3) 根据具体游戏优化启发式函数。

### Q: 搜索AI支持多人游戏吗？
A: 目前主要支持双人游戏，但可以通过扩展来支持多人游戏场景。

### Q: 如何添加自定义搜索算法？
A: 可以在 SearchAI 类中添加新的搜索方法，并在 `_snake_search_action` 等方法中调用。

## 版本历史

- **v1.0**: 初始版本，支持BFS、DFS、A*算法
- **当前**: 支持贪吃蛇、推箱子、五子棋等多种游戏

## 贡献

欢迎提交 Issue 和 Pull Request 来改进搜索算法AI的功能和性能。
