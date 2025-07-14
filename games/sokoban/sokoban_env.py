"""
推箱子游戏环境
实现gym风格的环境接口
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from games.base_env import BaseEnv
from games.sokoban.sokoban_game import SokobanGame


class SokobanEnv(BaseEnv):
    """推箱子游戏环境"""
    
    def __init__(self, level_id: int = 1, game_mode: str = 'competitive', **kwargs):
        """
        初始化推箱子环境
        
        Args:
            level_id: 关卡ID
            game_mode: 游戏模式 ('competitive': 竞争模式, 'cooperative': 合作模式)
        """
        self.level_id = level_id
        self.game_mode = game_mode
        
        # 创建游戏实例
        game = SokobanGame(level_id=level_id, game_mode=game_mode, **kwargs)
        super().__init__(game)
        
        # 环境特定属性
        self.max_episode_steps = kwargs.get('max_episode_steps', 500)
        self.reward_shaping = kwargs.get('reward_shaping', True)
        
    def _setup_spaces(self) -> None:
        """设置观察空间和动作空间"""
        # 观察空间：棋盘大小
        self.observation_space = {
            'shape': (self.game.height, self.game.width),
            'dtype': np.int32,
            'low': 0,
            'high': 8  # 最大符号值
        }
        
        # 动作空间：4个方向
        self.action_space = {
            'n': 4,
            'actions': ['UP', 'DOWN', 'LEFT', 'RIGHT']
        }
    
    def _get_observation(self) -> Dict[str, Any]:
        """获取观察"""
        state = self.game.get_state()
        
        # 基础观察：数字化的棋盘
        board_array = np.array(state['board_array'], dtype=np.int32)
        
        # 扩展观察信息
        observation = {
            'board': board_array,
            'player1_pos': np.array(state['player1_pos'] if state['player1_pos'] else [-1, -1], dtype=np.int32),
            'player2_pos': np.array(state['player2_pos'] if state['player2_pos'] else [-1, -1], dtype=np.int32),
            'current_player': state['current_player'],
            'player1_score': state['player1_score'],
            'player2_score': state['player2_score'],
            'boxes_left': state['total_boxes'] - state['completed_boxes'],
            'progress': state['completed_boxes'] / max(1, state['total_boxes']),
            'valid_actions_mask': self._get_action_mask()
        }
        
        return observation
    
    def _get_action_mask(self) -> np.ndarray:
        """获取动作掩码"""
        valid_actions = self.game.get_valid_actions()
        mask = np.zeros(4, dtype=bool)
        
        action_to_idx = {'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3}
        
        for action in valid_actions:
            if action in action_to_idx:
                mask[action_to_idx[action]] = True
        
        return mask
    
    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """重置环境"""
        self.game.reset()
        observation = self._get_observation()
        info = self._get_info()
        return observation, info
    
    def step(self, action) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """执行动作"""
        
        # 转换动作格式
        if isinstance(action, int):
            action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
            if 0 <= action < len(action_names):
                action = action_names[action]
            else:
                # 无效动作
                observation = self._get_observation()
                info = self._get_info()
                info['invalid_action'] = True
                return observation, -1.0, False, False, info
        
        # 检查动作是否有效
        valid_actions = self.game.get_valid_actions()
        
        if action not in valid_actions:
            # 无效动作的惩罚
            observation = self._get_observation()
            info = self._get_info()
            info['invalid_action'] = True
            return observation, -0.5, False, False, info
        # 执行动作
        _, reward, terminated, game_info = self.game.step(action)
        
        # 获取新的观察
        observation = self._get_observation()
        
        # 计算增强奖励
        if self.reward_shaping:
            reward = self._calculate_shaped_reward(reward, game_info)
        
        # 检查是否截断（达到最大步数）
        truncated = self.game.move_count >= self.max_episode_steps
        
        # 组合信息
        info = self._get_info()
        info.update(game_info)
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_shaped_reward(self, base_reward: float, game_info: Dict[str, Any]) -> float:
        """计算增强奖励"""
        reward = base_reward
        
        # 进度奖励
        progress = game_info.get('completed_boxes', 0) / max(1, len(self.game.targets))
        reward += progress * 0.1
        
        # 效率奖励（步数越少越好）
        current_player = self.game.current_player
        steps = self.game.player1_steps if current_player == 1 else self.game.player2_steps
        if steps > 0:
            efficiency_bonus = max(0, 1.0 - steps / 100.0) * 0.05
            reward += efficiency_bonus
        
        # 完成关卡的大奖励
        if self.game.is_terminal() and len(self.game.boxes_on_targets) == len(self.game.targets):
            reward += 50.0
        
        return reward
    
    def _get_info(self) -> Dict[str, Any]:
        """获取环境信息"""
        state = self.game.get_state()
        
        return {
            'level_id': self.level_id,
            'game_mode': self.game_mode,
            'current_player': state['current_player'],
            'player1_score': state['player1_score'],
            'player2_score': state['player2_score'],
            'player1_steps': state['player1_steps'],
            'player2_steps': state['player2_steps'],
            'total_boxes': state['total_boxes'],
            'completed_boxes': state['completed_boxes'],
            'progress': state['completed_boxes'] / max(1, state['total_boxes']),
            'move_count': state['move_count'],
            'is_terminal': state['is_terminal'],
            'winner': state['winner'],
            'valid_actions': self.game.get_valid_actions(),  # 直接调用而不是从状态获取
            'invalid_action': False
        }
    
    def render(self, mode: str = 'human') -> Optional[str]:
        """渲染环境"""
        if mode == 'human':
            print(self.game.render())
        elif mode == 'rgb_array':
            # 可以在这里实现图像渲染
            pass
        else:
            return self.game.render()
    
    def get_current_player(self) -> int:
        """获取当前玩家"""
        return self.game.current_player
    
    def get_winner(self) -> Optional[int]:
        """获取获胜者"""
        return self.game.get_winner()
    
    def get_game_state(self) -> Dict[str, Any]:
        """获取游戏状态"""
        return self.game.get_state()
    
    def clone(self) -> 'SokobanEnv':
        """克隆环境"""
        cloned_env = SokobanEnv(
            level_id=self.level_id,
            game_mode=self.game_mode,
            max_episode_steps=self.max_episode_steps,
            reward_shaping=self.reward_shaping
        )
        cloned_env.game = self.game.clone()
        return cloned_env
    
    def get_legal_actions(self) -> List[str]:
        """获取合法动作"""
        return self.game.get_valid_actions()
    
    def is_terminal(self) -> bool:
        """检查是否终止"""
        return self.game.is_terminal()
    
    def set_level(self, level_id: int):
        """设置关卡"""
        self.level_id = level_id
        self.game = SokobanGame(level_id=level_id, game_mode=self.game_mode)
        self._setup_spaces()
    
    def get_level_info(self) -> Dict[str, Any]:
        """获取关卡信息"""
        return {
            'level_id': self.level_id,
            'level_name': self.game.original_level.get('name', f'Level {self.level_id}'),
            'difficulty': self.game.original_level.get('difficulty', 'unknown'),
            'description': self.game.original_level.get('description', ''),
            'board_size': (self.game.height, self.game.width),
            'total_boxes': len(self.game.targets),
            'game_mode': self.game_mode
        }
    
    def get_available_levels(self) -> List[Dict[str, Any]]:
        """获取可用关卡列表"""
        levels = []
        for level_data in self.game.levels_data.get('levels', []):
            levels.append({
                'id': level_data['id'],
                'name': level_data.get('name', f"Level {level_data['id']}"),
                'difficulty': level_data.get('difficulty', 'unknown'),
                'description': level_data.get('description', '')
            })
        return levels
