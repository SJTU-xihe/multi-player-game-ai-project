"""
游戏模块
"""

from .base_game import BaseGame
from .base_env import BaseEnv

# 导入具体游戏
try:
    from .gomoku import GomokuGame, GomokuEnv
    __all__ = ['BaseGame', 'BaseEnv', 'GomokuGame', 'GomokuEnv']
except ImportError:
    __all__ = ['BaseGame', 'BaseEnv']

try:
    from .snake import SnakeGame, SnakeEnv
    __all__.extend(['SnakeGame', 'SnakeEnv'])
except ImportError:
    pass

try:
    from .sokoban import SokobanGame, SokobanEnv
    __all__.extend(['SokobanGame', 'SokobanEnv'])
except ImportError:
    pass 