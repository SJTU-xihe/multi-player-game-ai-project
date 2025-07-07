# 五子棋AI使用指南

## 概述

我你创建了一个专门针对五子棋游戏的智能AI：`GomokuMinimaxBot`，并且已经集成到图形界面中！这个AI使用了Minimax算法结合Alpha-Beta剪枝，专门为五子棋的规则和策略进行了优化。

## 🎮 图形界面使用方法（推荐）

### 快速开始
1. **启动游戏**: 运行 `python start_games.py`，选择选项1（多游戏GUI）
2. **选择五子棋**: 点击 "Gomoku" 按钮
3. **选择AI**: 点击 "Gomoku AI" 按钮（这就是新创建的专用五子棋AI）
4. **开始对战**: 点击 "New Game" 按钮
5. **下棋**: 用鼠标点击棋盘交叉点落子

### AI选择说明
在GUI界面中，你现在有以下AI选择：
- **Random AI**: 随机AI（适合练习）
- **Minimax AI**: 通用Minimax算法（中等难度）
- **MCTS AI**: 蒙特卡洛树搜索（高难度）
- **🆕 Gomoku AI**: 专门的五子棋AI（智能难度，推荐！）

## 文件位置

- **AI实现**: `agents/ai_bots/gomoku_minimax_bot.py`
- **测试文件**: `test_gomoku_ai.py`
- **简单测试**: `simple_test.py`
- **GUI集成测试**: `test_gui_integration.py`

## 五子棋AI的特点

### 1. 专门的评估函数
- **连五**: 100,000分（胜利）
- **活四**: 10,000分（必胜）
- **冲四**: 1,000分（需要堵）
- **活三**: 500分
- **眠三**: 50分
- **活二**: 10分
- **眠二**: 2分

### 2. 智能优化
- **迭代加深搜索**: 从浅层开始，逐步加深
- **Alpha-Beta剪枝**: 大幅减少搜索空间
- **状态缓存**: 避免重复计算
- **动作排序**: 优先搜索有希望的位置
- **超时控制**: 避免思考时间过长

### 3. 与贪吃蛇AI的区别

| 特性 | 贪吃蛇AI (minimax_bot.py) | 五子棋AI (gomoku_minimax_bot.py) |
|------|---------------------------|----------------------------------|
| **目标** | 存活并吃食物 | 连成五子获胜 |
| **评估重点** | 蛇长度、食物距离、空间控制 | 棋子连接模式、攻防平衡 |
| **动作空间** | 4方向移动 | 棋盘空位放置 |
| **状态特点** | 动态变化的蛇体 | 静态棋盘布局 |
| **策略** | 路径规划、避免碰撞 | 模式识别、攻防转换 |

## 使用方法

### 基本使用

```python
from agents.ai_bots.gomoku_minimax_bot import GomokuMinimaxBot
from games.gomoku.gomoku_env import GomokuEnv

# 创建环境
env = GomokuEnv(board_size=15)

# 创建AI
ai = GomokuMinimaxBot(
    name="智能五子棋AI", 
    player_id=1, 
    max_depth=4,    # 搜索深度
    timeout=5.0     # 最大思考时间（秒）
)

# 开始游戏
observation = env.reset()
action = ai.get_action(observation, env)
```

### 参数配置

- **max_depth**: 搜索深度（建议2-6，越高越强但越慢）
- **timeout**: 思考时间限制（秒）
- **player_id**: 玩家ID（1或2）

### 性能调优

```python
# 快速AI（适合实时对战）
fast_ai = GomokuMinimaxBot(max_depth=2, timeout=1.0)

# 强力AI（适合深度分析）
strong_ai = GomokuMinimaxBot(max_depth=6, timeout=10.0)

# 平衡AI（推荐设置）
balanced_ai = GomokuMinimaxBot(max_depth=4, timeout=5.0)
```

## 测试运行

运行以下命令测试AI：

```bash
# 激活虚拟环境
cd d:\Desktop\multi-player-game-ai-project
game_ai_env\Scripts\activate

# 运行简单测试
python simple_test.py

# 运行完整测试
python test_gomoku_ai.py
```

## AI策略说明

### 1. 开局策略
- 优先选择棋盘中心位置
- 在对手附近落子以增加交互

### 2. 中局策略
- 识别并形成有威胁的棋型
- 及时阻止对手的威胁
- 平衡攻守，寻找最佳时机

### 3. 终局策略
- 寻找必胜路径
- 防守对手的威胁棋型
- 最大化自己的胜率

### 4. 模式识别
AI能够识别以下五子棋模式：
- **活四**: `_●●●●_` (必胜)
- **冲四**: `●●●●_` 或 `●_●●●` (需要堵)
- **活三**: `_●●●_` (强威胁)
- **眠三**: `●●●__` (中等威胁)
- **活二**: `_●●_` (弱威胁)

## 与项目其他AI的集成

你可以在项目的主要文件中使用这个AI：

```python
# 在 start_games.py 或其他文件中
from agents.ai_bots import GomokuMinimaxBot

# 替换之前的AI
player1 = GomokuMinimaxBot(name="五子棋专家", player_id=1)
```

## 总结

现在你有了两个不同的AI：

1. **MinimaxBot** (`minimax_bot.py`): 专门为贪吃蛇游戏设计
2. **GomokuMinimaxBot** (`gomoku_minimax_bot.py`): 专门为五子棋游戏设计

每个AI都针对各自游戏的特点进行了优化，确保在相应的游戏中表现出色！
