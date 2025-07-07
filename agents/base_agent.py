"""
智能体基类
定义所有智能体的基本接口
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import time

class BaseAgent(ABC):
    """智能体基类"""
    def __init__(self, name="Agent", player_id=1):
        self.name = name
        self.player_id = player_id
        self.total_moves = 0
        self.total_time = 0.0

    @abstractmethod
    def get_action(self, observation, env):
        pass

    def reset(self):
        """重置智能体统计"""
        self.total_moves = 0
        self.total_time = 0.0
    
    def update_stats(self, result, time_taken):
        """更新智能体统计信息"""
        self.total_moves += 1
        self.total_time += time_taken
    
    def get_info(self):
        """获取智能体信息"""
        return {
            'name': self.name,
            'player_id': self.player_id,
            'type': self.__class__.__name__,
            'description': f'{self.__class__.__name__} 智能体',
            'total_moves': self.total_moves,
            'total_time': self.total_time,
            'avg_time_per_move': self.total_time / max(1, self.total_moves)
        }

    # ... 保持原有实现 ... 