"""
智能体模块
"""

from .base_agent import BaseAgent
from .human.human_agent import HumanAgent
from .ai_bots.random_bot import RandomBot
from .ai_bots.minimax_bot import MinimaxBot
from .ai_bots.mcts_bot import MCTSBot
from .ai_bots.rl_bot import RLBot
from .ai_bots.behavior_tree_bot import BehaviorTreeBot
from .ai_bots.snake_ai import SnakeAI, SmartSnakeAI

# 尝试导入推箱子AI
try:
    from .ai_bots.sokoban_ai import SokobanAI, SimpleSokobanAI
    sokoban_imported = True
except ImportError:
    sokoban_imported = False

__all__ = [
    'BaseAgent',
    'HumanAgent',
    'RandomBot',
    'MinimaxBot',
    'MCTSBot',
    'RLBot',
    'BehaviorTreeBot',
    'SnakeAI',
    'SmartSnakeAI'
]

if sokoban_imported:
    __all__.extend(['SokobanAI', 'SimpleSokobanAI']) 