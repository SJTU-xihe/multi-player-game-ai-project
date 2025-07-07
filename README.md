# 多人游戏AI框架

一个基于OpenAI Gym风格的多人游戏AI对战框架，支持五子棋和贪吃蛇游戏，提供图形界面和命令行两种模式。


### 1. conda 虚拟环境  (推荐)
在conda提供的命令行界面里键入命令:

```bash
conda create --name MultiPlayerGame python=3.11  # 创建虚拟环境, 推荐python版本为3.11

# 激活虚拟环境
conda activate MultiPlayerGame
# 升级pip
python -m pip install --upgrade pip
```

### 2. 项目clone
```bash
# clone 项目 (推荐使用Git, 当然你也可以直接download下来)
git clone https://github.com/ying-wen/multi-player-game-ai-project
# 进入项目目录
cd multi-player-game-ai-project
```

### 4. python 依赖安装

```bash
# 安装项目依赖
pip install -r requirements.txt
```


#### Linux
```bash
# 安装图形界面支持
# Ubuntu/Debian:
sudo apt install python3-tk #如果使用虚拟环境, tk 也可以使用 pip install tk 安装在python虚拟环境下, 这里是给系统的python解释器安装 pip 包

# 如果使用ssh连接远程服务器，启用X11转发可以在本地显示图形界面
ssh -X username@hostname
```

## 🚀 快速启动

### 验证安装
```bash
# 检查Python版本
python --version

# 检查pip版本
pip --version

# 检查依赖安装
python -c "import pygame, numpy; print('Dependencies OK')"
```

### 启动项目
```bash
# 最简单的启动方式
python start_games.py
```

然后根据菜单选择：
- **选择1**: 多游戏GUI (五子棋+贪吃蛇)
- **选择2**: 贪吃蛇专用GUI (推荐贪吃蛇玩家)
- **选择3**: 五子棋命令行版本
- **选择4**: 贪吃蛇命令行版本
- **选择5**: 运行测试
- **选择6**: 退出


## 🎮 支持的游戏

### 1. 五子棋 (Gomoku)
- **规则**: 15×15棋盘，连成5子获胜
- **操作**: 图形界面点击落子，命令行输入坐标
- **AI支持**: 随机AI、Minimax算法、MCTS算法

### 2. 贪吃蛇 (Snake)
- **规则**: 双人贪吃蛇对战，吃食物长大，避免碰撞
- **操作**: 方向键或WASD控制移动
- **AI支持**: 基础贪吃蛇AI、智能寻路AI

### 3. 推箱子 (Sokoban)
- **规则**: 推动箱子到目标位置，智能解谜游戏
- **操作**: 方向键或WASD控制移动和推箱子
- **AI支持**: 智能推箱子AI、简单推箱子AI
- **关卡系统**: 支持多关卡切换，逐步增加难度

## 🎯 图形界面说明

### 1. 多游戏GUI (`gui_game.py`)
**特点**: 支持五子棋、贪吃蛇和推箱子切换，全功能界面
- 🖱️ **五子棋**: 鼠标点击落子
- ⌨️ **贪吃蛇**: 方向键/WASD控制
- 📦 **推箱子**: 方向键/WASD控制，支持关卡切换
- 🎮 **游戏切换**: 点击按钮切换游戏类型
- 🤖 **AI选择**: 随机AI、MinimaxAI、MCTS AI、🔍**搜索AI**(BFS/DFS/A*)
- ⏸️ **暂停功能**: 随时暂停/继续游戏

> 🔍 **新增搜索AI**: 在AI选择区域新增了"Search BFS"、"Search DFS"、"Search A*"选项，体验不同搜索算法的游戏策略！详见 [GUI搜索AI更新说明](GUI_SEARCH_AI_UPDATE.md)

### 3. 推箱子专用GUI (`sokoban_gui.py`)
**特点**: 专为推箱子优化，完整的解谜体验
- � **专用界面**: 针对推箱子优化的UI设计
- 🎨 **视觉效果**: 清晰的箱子、目标点、玩家显示
- 🔄 **关卡切换**: 支持多关卡前后切换
- 🤖 **专用AI**: 智能AI、简单AI、随机AI
- 📊 **实时信息**: 关卡进度、移动步数实时显示

### 4. 推箱子关卡编辑器 (`sokoban_editor.py`)
**特点**: 可视化的关卡设计工具
- 🛠️ **拖拽编辑**: 鼠标点击放置游戏元素
- 💾 **保存加载**: 关卡数据保存和加载功能
- ✅ **关卡验证**: 自动检查关卡设计的合理性
- 🧪 **测试功能**: 可直接测试设计的关卡

## 🎮 游戏操作指南

### 五子棋操作
- **目标**: 连成5子获胜
- **操作**: 鼠标点击棋盘交叉点落子
- **显示**: 黑子是你，白子是AI
- **标记**: 红圈标记最后一步

### 推箱子操作
- **目标**: 推动所有箱子到目标位置
- **操作**: 
  - 方向键 ↑↓←→ 控制移动
  - 或者 WASD 键控制移动
  - 靠近箱子时推动箱子
- **显示**: 
  - 蓝色圆圈是玩家1
  - 红色圆圈是玩家2 
  - 棕色方块是箱子
  - 粉色标记是目标点
  - 橙色方块是已放置到位的箱子
- **关卡切换**: 点击Prev/Next按钮切换关卡
- **获胜**: 所有箱子都推到目标位置

## 🧠 AI算法说明

### 通用AI
- **RandomBot**: 完全随机动作，适合练习
- **HumanAgent**: 人类玩家接口

### 五子棋AI
- **MinimaxBot**: 经典博弈树搜索，中等难度
- **MCTSBot**: 蒙特卡洛树搜索，高难度

### 推箱子AI
- **SokobanAI**: 智能AI，使用A*搜索和启发式策略
- **SimpleSokobanAI**: 简单AI，基础推箱子策略

### 🔍 搜索算法AI (新增)
- **SearchAI**: 通用搜索算法框架，支持多种搜索策略
  - **BFS (广度优先搜索)**: 保证找到最短路径解
  - **DFS (深度优先搜索)**: 内存效率高，适合深度探索
  - **A* (启发式搜索)**: 结合启发式函数的最优搜索
- **适用游戏**: 支持五子棋、贪吃蛇、推箱子等多种游戏

### 推箱子AI
- **SokobanAI**: 智能AI，使用A*搜索和启发式策略
- **SimpleSokobanAI**: 简单AI，基础推箱子策略

### 🔍 搜索算法AI (新增)
- **SearchAI**: 通用搜索算法框架，支持多种搜索策略
  - **BFS (广度优先搜索)**: 保证找到最短路径解
  - **DFS (深度优先搜索)**: 内存效率高，适合深度探索
  - **A* (启发式搜索)**: 结合启发式函数的最优搜索
- **适用游戏**: 支持五子棋、贪吃蛇、推箱子等多种游戏

## 💻 命令行模式

### 直接启动命令
```bash
# 五子棋人机对战
python main.py --game gomoku --player1 human --player2 minimax

# 贪吃蛇人机对战
python main.py --game snake --player1 human --player2 snake_ai

# AI对战观看
python main.py --game gomoku --player1 mcts --player2 minimax

# 🔍 新增：搜索算法AI对战
python main.py --game snake --player1 search_bfs --player2 search_astar
python main.py --game gomoku --player1 search_astar --player2 random
python main.py --game snake --player1 search_dfs --player2 random --no-render
```

### 可用智能体
- `human`: 人类玩家
- `random`: 随机AI
- `minimax`: Minimax算法AI
- `mcts`: MCTS算法AI
- `snake_ai`: 贪吃蛇基础AI
- `smart_snake_ai`: 贪吃蛇智能AI
- `search`: 搜索算法AI (默认BFS)
- `search_bfs`: 广度优先搜索AI
- `search_dfs`: 深度优先搜索AI
- `search_astar`: A*搜索算法AI

> 🔍 **新增搜索算法AI**: 实现了BFS、DFS、A*等经典搜索算法，支持贪吃蛇、推箱子、五子棋等多种游戏。详见 [搜索AI使用指南](SEARCH_AI_README.md)

## 📦 依赖说明

### 核心依赖
```txt
pygame>=2.1.0       # 图形界面和游戏引擎
numpy>=1.19.0       # 数值计算和数组操作
typing-extensions   # 类型提示支持
```

### 可选依赖
```bash
# 开发和测试
pytest              # 单元测试框架
Ruff               # 代码格式化(建议vscode安装插件 'Ruff' )
flake8              # 代码风格检查

# 性能分析
cProfile            # Python内置性能分析器
memory_profiler     # 内存使用分析
```
## 🧪 测试验证

### 运行完整测试
```bash
python test_project.py
```

**测试结果**: 所有测试通过 (7/7)
- ✅ 模块导入测试
- ✅ 五子棋游戏逻辑测试
- ✅ 五子棋环境测试
- ✅ AI智能体测试
- ✅ 游戏对战测试
- ✅ 智能体评估测试
- ✅ 自定义智能体测试

### 单元测试
```bash
# 运行特定测试
python -m pytest tests/

# 运行覆盖率测试
python -m pytest --cov=games --cov=agents tests/
```

## 🛠️ 项目结构

```
multi-player-game-ai-project/
├── gui_game.py           # 多游戏图形界面
├── snake_gui.py          # 贪吃蛇专用GUI
├── sokoban_gui.py        # 推箱子专用GUI
├── sokoban_editor.py     # 推箱子关卡编辑器
├── start_games.py        # 启动脚本
├── main.py               # 命令行主程序
├── test_project.py       # 测试程序
├── test_search_ai.py     # 搜索AI测试
├── config.py             # 配置文件
├── requirements.txt      # 依赖列表
├── BUG_FIXES_HISTORY.md  # Bug修复历史记录
├── SEARCH_AI_README.md   # 搜索AI使用指南
├── GUI_SEARCH_AI_UPDATE.md # GUI搜索AI更新说明
├── .gitignore           # Git忽略文件
├── games/                # 游戏模块
│   ├── __init__.py
│   ├── base_game.py     # 游戏基类
│   ├── base_env.py      # 环境基类
│   ├── gomoku/          # 五子棋
│   │   ├── __init__.py
│   │   ├── gomoku_game.py
│   │   └── gomoku_env.py
│   ├── snake/           # 贪吃蛇
│   │   ├── __init__.py
│   │   ├── snake_game.py
│   │   └── snake_env.py
│   └── sokoban/         # 推箱子
│       ├── __init__.py
│       ├── sokoban_game.py
│       ├── sokoban_env.py
│       └── levels.json  # 关卡数据
├── agents/              # AI智能体
│   ├── __init__.py
│   ├── base_agent.py    # 智能体基类
│   ├── human/           # 人类智能体
│   │   ├── __init__.py
│   │   ├── human_agent.py
│   │   └── gui_human_agent.py
│   └── ai_bots/         # AI机器人
│       ├── __init__.py
│       ├── random_bot.py
│       ├── minimax_bot.py
│       ├── mcts_bot.py
│       ├── rl_bot.py
│       ├── behavior_tree_bot.py
│       ├── snake_ai.py
│       ├── sokoban_ai.py      # 推箱子AI
│       ├── gomoku_minimax_bot.py # 五子棋专用AI
│       └── search_ai.py       # 搜索算法AI (新增)
├── utils/               # 工具模块
│   ├── __init__.py
│   └── game_utils.py
├── examples/            # 示例代码
│   ├── basic_usage.py
│   ├── custom_agent.py
│   └── search_ai_examples.py  # 搜索AI示例 (新增)
└── tests/               # 测试文件
    └── __init__.py
```

## 🎯 使用建议

### 新手推荐
1. **开始**: 运行 `python start_games.py`
2. **五子棋**: 选择1，然后选择随机AI练习
3. **贪吃蛇**: 选择2，体验专用界面

### 进阶玩家
1. **挑战高难度**: 选择MCTS AI对战
2. **观察AI**: 使用命令行模式观看AI对战
3. **自定义**: 修改AI参数或添加新算法

### 开发者
1. **测试**: 先运行 `python test_project.py`
2. **扩展**: 参考现有代码添加新游戏或AI
3. **调试**: 使用命令行模式便于调试

## 🐛 故障排除

### 环境问题

**Q: Python版本不兼容？**
A: 确保使用Python 3.7+，推荐3.8-3.11版本

**Q: pip安装失败？**
A: 
```bash
# 升级pip
python -m pip install --upgrade pip

# 使用国内镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 使用conda
conda install pygame numpy
```

**Q: 虚拟环境问题？**
A:
```bash
# 删除旧环境
conda env remove --name your_env_name

# 重新创建
conda create --name MultiPlayerGame python=3.11  # 创建虚拟环境, 推荐python版本为3.11

pip install -r requirements.txt
```

### 图形界面问题

**Q: pygame窗口无法显示？**
A: 
- **Windows**: 检查是否安装了Visual C++ Redistributable
- **macOS**: 安装XQuartz (`brew install --cask xquartz`)
- **Linux**: 安装图形界面支持 (`sudo apt install python3-tk`)
- **虚拟环境** `pip install tk`
**Q: SSH远程连接无法显示图形？**
A:
```bash
# 启用X11转发
ssh -X username@hostname

# 或者使用VNC/远程桌面
```

**Q: 中文显示乱码？**
A: 项目已使用英文界面，避免了字体问题

tip:在大多数情况下请尽量使用英文命名你的任何工作， 可以解决许多潜在的问题
### 游戏问题

**Q: 贪吃蛇移动太快/太慢？**
A: 修改配置文件中的 `update_interval` 参数

**Q: AI思考时间太长？**
A: 调整AI参数：
- Minimax: 减少 `max_depth`
- MCTS: 减少 `simulation_count`

**Q: 导入错误？**
A:
```bash
# 确保在项目根目录
cd multi-player-game-ai-project

# 检查Python路径
python -c "import sys; print(sys.path)"

# 设置PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### 性能优化

**Q: 游戏运行卡顿？**
A:
- 关闭不必要的后台程序
- 降低AI搜索深度
- 使用更快的硬件

**Q: 内存使用过高？**
A:
- 减少MCTS模拟次数
- 定期清理游戏历史记录
- 使用更轻量的AI算法

## 🔧 高级配置

### 自定义配置
编辑 `config.py` 文件可以调整：
```python
# 游戏参数
BOARD_SIZE = 15          # 棋盘大小
WIN_LENGTH = 5           # 获胜条件
SNAKE_SPEED = 0.3        # 贪吃蛇速度

# AI参数
MINIMAX_DEPTH = 3        # Minimax搜索深度
MCTS_SIMULATIONS = 500   # MCTS模拟次数

# 界面参数
WINDOW_WIDTH = 800       # 窗口宽度
WINDOW_HEIGHT = 600      # 窗口高度
```

### 添加新游戏
1. 在 `games/` 目录下创建新游戏文件夹
2. 继承 `BaseGame` 和 `BaseEnv` 基类
3. 实现必要的方法
4. 在 `main.py` 中注册新游戏

### 添加新AI
1. 在 `agents/ai_bots/` 目录下创建新AI文件
2. 继承 `BaseAgent` 基类
3. 实现 `get_action` 方法
4. 在 `agents/__init__.py` 中导入

## 🎊 项目亮点

### 完成的功能
- ✅ **三游戏支持**: 五子棋、贪吃蛇和推箱子
- ✅ **多种GUI**: 主GUI、专用GUI和关卡编辑器
- ✅ **丰富AI**: 10+种不同算法的AI，包含搜索算法AI
- ✅ **人机对战**: 流畅的实时对战体验
- ✅ **关卡系统**: 推箱子支持多关卡和关卡编辑
- ✅ **命令行模式**: 便于开发和调试
- ✅ **完整测试**: 所有功能经过验证
- ✅ **用户友好**: 简单的启动和操作

### 技术特色
- 🏗️ **模块化设计**: 易于扩展新游戏和AI
- 🎯 **Gym风格**: 标准化的环境接口
- 🔍 **搜索算法**: 完整的BFS/DFS/A*搜索算法实现
- 📦 **多游戏架构**: 统一接口支持不同类型游戏
- 🎮 **GUI集成**: 搜索AI无缝集成到图形界面
- 🧪 **测试驱动**: 完整的测试覆盖
- 📚 **文档完善**: 详细的使用说明

## 📋 作业要求

### 基本要求 
1. **修复项目错误** 
   - [x] 修复导入错误
   - [x] 修复语法错误
   - [x] 确保所有测试通过

2. **完善AI Bot** 
   - [x] 检查MinimaxBot的完整逻辑
   - [x] 检查完善MCTSBot的蒙特卡洛树搜索
   - [x] 检查完善贪吃蛇专用AI

3. **测试和验证** 
   - [x] 所有测试用例通过
   - [x] AI对战功能正常
   - [x] 人机对战功能正常
   - [x] 图形界面正常

### 扩展要求
1. **实现至少一个新游戏** 
   - [x] 至少支持双人对战模式
   - [x] 支持图形界面

2. **实现新AI Bot** 

### 额外功能
- [x] **图形界面**: 完整的pygame图形界面
- [x] **多游戏支持**: 在同一界面切换不同游戏
- [x] **实时对战**: 流畅的人机对战体验
- [x] **暂停功能**: 游戏过程中可暂停/继续
- [x] **启动脚本**: 用户友好的启动方式

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

### Github + git 的开发指南(建议预先学习git)
1. 在github上 Fork 项目
2. git clone到本地
3. 在本地创建特性分支 (`git checkout -b feature/AmazingFeature`)
4. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
5. 推送到分支 (`git push origin feature/AmazingFeature`)
6. 打开Pull Request

## 📄 许可证

MIT License

