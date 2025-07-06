"""
环境基类
实现gym风格的游戏环境接口
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import config
from games.base_game import BaseGame

class BaseEnv(ABC):
    """环境基类，实现gym风格接口"""
    
    def __init__(self, game: BaseGame):
        self.game = game
        self.observation_space = None
        self.action_space = None
        self._setup_spaces()
    
    @abstractmethod
    def _setup_spaces(self) -> None:
        """设置观察空间和动作空间"""
        pass
    
    @abstractmethod
    def _get_observation(self) -> np.ndarray:
        """获取观察"""
        pass
    
    @abstractmethod
    def _get_action_mask(self) -> np.ndarray:
        """获取动作掩码"""
        pass
    
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置环境"""
        self.game.reset()
        observation = self._get_observation()
        info = self.game.get_game_info()
        return observation, info
    
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """执行动作"""
        # 检查动作是否有效
        valid_actions = self.game.get_valid_actions(self.game.current_player)
        if action not in valid_actions:
            # 对于贪吃蛇游戏，无效动作（如反向移动）应该让当前玩家失败
            if hasattr(self.game, 'handle_invalid_action'):
                observation, reward, done, info = self.game.handle_invalid_action(action)
                truncated = self.game.is_timeout()
                return observation, reward, done, truncated, info
            else:
                return self._get_observation(), -1000, True, False, {'error': 'Invalid action'}
        
        # 执行动作
        observation, reward, done, info = self.game.step(action)
        
        # 更新游戏状态
        self.game.update_game_state()
        
        # 检查是否超时
        truncated = self.game.is_timeout()
        
        return observation, reward, done, truncated, info
    
    def render(self, mode='human') -> Optional[np.ndarray]:
        """渲染环境"""
        if mode == 'human':
            return self.game.render()
        elif mode == 'rgb_array':
            return self._get_observation()
        return None
    
    def close(self) -> None:
        """关闭环境"""
        pass
    
    def get_valid_actions(self) -> List[Any]:
        """获取有效动作"""
        return self.game.get_valid_actions(self.game.current_player)
    
    def get_action_mask(self) -> np.ndarray:
        """获取动作掩码"""
        return self._get_action_mask() 