"""
推箱子游戏逻辑
实现双人对战推箱子游戏
"""

import json
import os
import copy
import time
from typing import Dict, List, Tuple, Any, Optional, Set
import numpy as np
from games.base_game import BaseGame
import config


class SokobanGame(BaseGame):
    """推箱子游戏类 - 双人对战版本"""
    
    # 游戏符号
    WALL = '#'
    FLOOR = ' '
    BOX = '$'
    TARGET = '.'
    BOX_ON_TARGET = '*'
    PLAYER1 = '@'  # 玩家1
    PLAYER2 = '&'  # 玩家2  
    PLAYER1_ON_TARGET = '+'
    PLAYER2_ON_TARGET = '%'
    
    # 移动方向
    DIRECTIONS = {
        'UP': (-1, 0),
        'DOWN': (1, 0),
        'LEFT': (0, -1),
        'RIGHT': (0, 1)
    }
    
    def __init__(self, level_id: int = 1, game_mode: str = 'competitive', **kwargs):
        """
        初始化推箱子游戏
        
        Args:
            level_id: 关卡ID
            game_mode: 游戏模式 ('competitive': 竞争模式, 'cooperative': 合作模式)
        """
        self.level_id = level_id
        self.game_mode = game_mode  # 'competitive' or 'cooperative'
        self.max_steps = kwargs.get('max_steps', 500)
        self.time_limit = kwargs.get('time_limit', 300)  # 5分钟
        
        # 加载关卡数据
        self.levels_data = self._load_levels()
        self.original_level = self._get_level(level_id)
        
        # 游戏状态
        self.board = None
        self.player1_pos = None
        self.player2_pos = None
        self.targets = set()
        self.boxes = set()
        self.boxes_on_targets = set()
        
        # 分数和统计
        self.player1_score = 0  # 玩家1推入目标的箱子数
        self.player2_score = 0  # 玩家2推入目标的箱子数
        self.player1_steps = 0
        self.player2_steps = 0
        
        super().__init__(kwargs)
    
    def _load_levels(self) -> Dict:
        """加载关卡数据"""
        levels_file = os.path.join(os.path.dirname(__file__), 'levels.json')
        try:
            with open(levels_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            # 如果文件不存在，返回默认关卡
            return {
                "levels": [{
                    "id": 1,
                    "name": "默认关卡",
                    "map": [
                        "########",
                        "#      #",
                        "# $@   #",
                        "#    . #",
                        "#      #",
                        "########"
                    ]
                }]
            }
    
    def _get_level(self, level_id: int) -> Dict:
        """获取指定关卡"""
        for level in self.levels_data.get('levels', []):
            if level['id'] == level_id:
                return level
        
        # 如果找不到指定关卡，返回第一个关卡
        return self.levels_data['levels'][0] if self.levels_data.get('levels') else {
            "id": 1, "name": "默认", "map": ["####", "#@.#", "####"]
        }
    
    def reset(self) -> Dict[str, Any]:
        """重置游戏状态"""
        # 重置基类状态
        self.current_player = 1
        self.game_state = config.GameState.ONGOING
        self.move_count = 0
        self.start_time = time.time()
        self.last_move_time = time.time()
        self.history = []
        
        # 解析关卡地图
        level_map = self.original_level['map']
        self.height = len(level_map)
        self.width = max(len(row) for row in level_map)
        
        # 初始化棋盘
        self.board = [[self.FLOOR for _ in range(self.width)] for _ in range(self.height)]
        self.targets = set()
        self.boxes = set()
        self.boxes_on_targets = set()
        
        # 解析地图并放置元素
        player_count = 0
        for row in range(self.height):
            line = level_map[row] if row < len(level_map) else ""
            for col in range(len(line)):
                char = line[col]
                
                if char == self.WALL:
                    self.board[row][col] = self.WALL
                elif char == self.TARGET:
                    self.targets.add((row, col))
                    self.board[row][col] = self.TARGET
                elif char == self.BOX:
                    self.boxes.add((row, col))
                    self.board[row][col] = self.BOX
                elif char == self.BOX_ON_TARGET:
                    self.targets.add((row, col))
                    self.boxes.add((row, col))
                    self.boxes_on_targets.add((row, col))
                    self.board[row][col] = self.BOX_ON_TARGET
                elif char == '@':  # 原始玩家位置
                    if player_count == 0:
                        self.player1_pos = (row, col)
                        self.board[row][col] = self.PLAYER1
                        player_count += 1
                    else:
                        self.player2_pos = (row, col)
                        self.board[row][col] = self.PLAYER2
                        player_count += 1
                elif char == '&':  # 玩家2位置
                    self.player2_pos = (row, col)
                    self.board[row][col] = self.PLAYER2
                    player_count += 1
                elif char == self.PLAYER1_ON_TARGET:
                    self.player1_pos = (row, col)
                    self.targets.add((row, col))
                    self.board[row][col] = self.PLAYER1_ON_TARGET
                    player_count += 1
        
        # 如果只有一个玩家位置，为第二个玩家找一个空位置
        if not self.player2_pos:
            self.player2_pos = self._find_empty_position()
            if self.player2_pos:
                row, col = self.player2_pos
                if (row, col) in self.targets:
                    self.board[row][col] = self.PLAYER2_ON_TARGET
                else:
                    self.board[row][col] = self.PLAYER2
        
        # 强制检查和修复玩家位置冲突
        if self.player1_pos == self.player2_pos:
            self._fix_initial_player_positions()
        
        # 重置分数和统计
        self.player1_score = 0
        self.player2_score = 0
        self.player1_steps = 0
        self.player2_steps = 0
        
        return self.get_state()
    
    def _find_empty_position(self) -> Optional[Tuple[int, int]]:
        """找到一个空的位置放置第二个玩家"""
        for row in range(self.height):
            for col in range(self.width):
                pos = (row, col)
                # 确保这个位置不是玩家1的位置，也不是墙壁或箱子
                if (pos != self.player1_pos and 
                    (self.board[row][col] == self.FLOOR or 
                     self.board[row][col] == self.TARGET) and
                    pos not in self.boxes):
                    return pos
        return None
    
    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        执行一步动作
        
        Args:
            action: 动作 ('UP', 'DOWN', 'LEFT', 'RIGHT')
            
        Returns:
            observation: 游戏状态
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        if self.is_terminal():
            return self.get_state(), 0, True, {'reason': 'Game already ended'}
        
        # 记录移动前的状态
        old_score = self.player1_score if self.current_player == 1 else self.player2_score
        
        # 执行移动
        success, push_result = self._move_player(self.current_player, action)
        
        # 计算奖励
        reward = self._calculate_reward(success, push_result, old_score)
        
        # 更新步数
        if self.current_player == 1:
            self.player1_steps += 1
        else:
            self.player2_steps += 1
        
        # 记录移动
        self.record_move(self.current_player, action, {
            'success': success,
            'push_result': push_result,
            'reward': reward
        })
        
        # 检查游戏是否结束
        done = self.is_terminal()
        
        if not done:
            # 切换玩家（如果是竞争模式）
            if self.game_mode == 'competitive':
                self.switch_player()
        
        info = {
            'success': success,
            'push_result': push_result,
            'current_player': self.current_player,
            'player1_score': self.player1_score,
            'player2_score': self.player2_score,
            'boxes_left': len(self.boxes) - len(self.boxes_on_targets)
        }
        
        return self.get_state(), reward, done, info
    
    def _move_player(self, player: int, direction: str) -> Tuple[bool, str]:
        """
        移动玩家
        
        Returns:
            success: 是否移动成功
            push_result: 推箱结果 ('none', 'success', 'blocked')
        """
        if direction not in self.DIRECTIONS:
            return False, 'none'
        
        # 获取当前玩家位置
        current_pos = self.player1_pos if player == 1 else self.player2_pos
        if not current_pos:
            return False, 'none'
        
        # 计算新位置
        dr, dc = self.DIRECTIONS[direction]
        new_row, new_col = current_pos[0] + dr, current_pos[1] + dc
        
        # 检查边界
        if not self._is_valid_position(new_row, new_col):
            return False, 'none'
        
        # 检查目标位置
        target_cell = self.board[new_row][new_col]
        
        # 如果目标位置是墙壁，不能移动
        if target_cell == self.WALL:
            return False, 'none'
        
        # 如果目标位置是另一个玩家，不能移动
        other_player_pos = self.player2_pos if player == 1 else self.player1_pos
        if other_player_pos and (new_row, new_col) == other_player_pos:
            return False, 'none'
        
        # 特殊处理：如果两个玩家在同一位置，允许其中一个移动
        if self.player1_pos == self.player2_pos and current_pos == (new_row, new_col):
            return False, 'none'
        
        # 如果目标位置有箱子，尝试推箱子
        if (new_row, new_col) in self.boxes:
            push_result = self._push_box(new_row, new_col, direction, player)
            if push_result == 'blocked':
                return False, 'blocked'
        else:
            push_result = 'none'
        
        # 移动玩家
        self._update_player_position(player, current_pos, (new_row, new_col))
        
        return True, push_result
    
    def _push_box(self, box_row: int, box_col: int, direction: str, player: int) -> str:
        """
        推箱子
        
        Returns:
            'success': 推箱成功
            'blocked': 推箱被阻挡
        """
        dr, dc = self.DIRECTIONS[direction]
        new_box_row, new_box_col = box_row + dr, box_col + dc
        
        # 检查箱子新位置是否有效
        if not self._is_valid_position(new_box_row, new_box_col):
            return 'blocked'
        
        # 检查箱子新位置是否被占用
        target_cell = self.board[new_box_row][new_box_col]
        if (target_cell == self.WALL or 
            (new_box_row, new_box_col) in self.boxes or
            (new_box_row, new_box_col) == self.player1_pos or
            (new_box_row, new_box_col) == self.player2_pos):
            return 'blocked'
        
        # 移动箱子
        self.boxes.remove((box_row, box_col))
        self.boxes.add((new_box_row, new_box_col))
        
        # 更新箱子在目标上的状态
        was_on_target = (box_row, box_col) in self.boxes_on_targets
        now_on_target = (new_box_row, new_box_col) in self.targets
        
        if was_on_target:
            self.boxes_on_targets.remove((box_row, box_col))
        
        if now_on_target:
            self.boxes_on_targets.add((new_box_row, new_box_col))
            # 增加玩家分数
            if player == 1:
                self.player1_score += 1
            else:
                self.player2_score += 1
        elif was_on_target:
            # 从目标上移开了箱子，减少分数
            if player == 1:
                self.player1_score = max(0, self.player1_score - 1)
            else:
                self.player2_score = max(0, self.player2_score - 1)
        
        # 更新棋盘显示
        self._update_board_display()
        
        return 'success'
    
    def _update_player_position(self, player: int, old_pos: Tuple[int, int], new_pos: Tuple[int, int]):
        """更新玩家位置"""
        old_row, old_col = old_pos
        new_row, new_col = new_pos
        
        # 更新位置记录
        if player == 1:
            self.player1_pos = new_pos
        else:
            self.player2_pos = new_pos
        
        # 更新棋盘显示
        self._update_board_display()
    
    def _update_board_display(self):
        """更新棋盘显示"""
        # 重置棋盘（保留墙壁和目标）
        for row in range(self.height):
            for col in range(self.width):
                if self.board[row][col] not in [self.WALL]:
                    if (row, col) in self.targets:
                        self.board[row][col] = self.TARGET
                    else:
                        self.board[row][col] = self.FLOOR
        
        # 放置箱子
        for box_pos in self.boxes:
            row, col = box_pos
            if box_pos in self.targets:
                self.board[row][col] = self.BOX_ON_TARGET
            else:
                self.board[row][col] = self.BOX
        
        # 放置玩家（确保位置不冲突）
        if self.player1_pos:
            row, col = self.player1_pos
            # 检查是否与玩家2位置冲突
            if self.player2_pos and self.player1_pos == self.player2_pos:
                # 强制分离玩家位置
                self._fix_player_position_conflict()
                return
            
            if (row, col) in self.targets:
                self.board[row][col] = self.PLAYER1_ON_TARGET
            else:
                self.board[row][col] = self.PLAYER1
        
        if self.player2_pos:
            row, col = self.player2_pos
            if (row, col) in self.targets:
                self.board[row][col] = self.PLAYER2_ON_TARGET
            else:
                self.board[row][col] = self.PLAYER2
    
    def _is_valid_position(self, row: int, col: int) -> bool:
        """检查位置是否有效"""
        return 0 <= row < self.height and 0 <= col < self.width
    
    def _calculate_reward(self, success: bool, push_result: str, old_score: int) -> float:
        """计算奖励"""
        reward = 0
        
        if not success:
            reward = -0.1  # 无效移动的小惩罚
        else:
            reward = 0.01  # 有效移动的小奖励
            
            if push_result == 'success':
                current_score = self.player1_score if self.current_player == 1 else self.player2_score
                if current_score > old_score:
                    reward += 10  # 推箱子到目标的大奖励
                else:
                    reward += 1   # 推箱子的小奖励
        
        return reward
    
    def get_valid_actions(self, player: int = None) -> List[str]:
        """获取有效动作列表"""
        if self.is_terminal():
            return []
        
        valid_actions = []
        current_player = player if player is not None else self.current_player
        current_pos = self.player1_pos if current_player == 1 else self.player2_pos
        
        if not current_pos:
            return []
        
        for direction in self.DIRECTIONS:
            dr, dc = self.DIRECTIONS[direction]
            new_row, new_col = current_pos[0] + dr, current_pos[1] + dc
            
            # 基本边界检查
            if not self._is_valid_position(new_row, new_col):
                continue
            
            # 检查是否是墙壁
            if self.board[new_row][new_col] == self.WALL:
                continue
            
            # 检查是否是另一个玩家
            other_player_pos = self.player2_pos if current_player == 1 else self.player1_pos
            # 特殊处理：如果两个玩家在同一位置，只有移动方向不是对方才被阻挡
            if (other_player_pos and (new_row, new_col) == other_player_pos and 
                current_pos != other_player_pos):  # 只有当两个玩家不在同一位置时才检查碰撞
                continue
            
            # 如果是箱子，检查是否能推
            if (new_row, new_col) in self.boxes:
                box_dr, box_dc = dr, dc
                box_new_row, box_new_col = new_row + box_dr, new_col + box_dc
                
                # 检查箱子能否被推到新位置
                if (self._is_valid_position(box_new_row, box_new_col) and
                    self.board[box_new_row][box_new_col] != self.WALL and
                    (box_new_row, box_new_col) not in self.boxes and
                    (box_new_row, box_new_col) != self.player1_pos and
                    (box_new_row, box_new_col) != self.player2_pos):
                    valid_actions.append(direction)
            else:
                valid_actions.append(direction)
        
        return valid_actions
    
    def is_terminal(self) -> bool:
        """检查游戏是否结束"""
        # 检查是否所有箱子都在目标上（胜利条件）
        if len(self.boxes_on_targets) == len(self.targets):
            return True
        
        # 检查是否有箱子陷入死锁（失败条件）
        if self._check_deadlock():
            return True
        
        # 检查是否超时
        if self.is_timeout():
            return True
        
        # 检查是否达到最大步数
        if self.move_count >= self.max_steps:
            return True
        
        return False
    
    def _check_deadlock(self) -> bool:
        """检查是否存在死锁情况"""
        # 检查每个不在目标上的箱子
        for box_pos in self.boxes:
            if box_pos not in self.targets:
                if self._is_box_deadlocked(box_pos):
                    return True
        return False
    
    def _is_box_deadlocked(self, box_pos: Tuple[int, int]) -> bool:
        """检查单个箱子是否死锁"""
        row, col = box_pos
        
        # 检查角落死锁
        if self._is_corner_deadlock(row, col):
            return True
        
        # 检查边缘死锁
        if self._is_edge_deadlock(row, col):
            return True
        
        return False
    
    def _is_corner_deadlock(self, row: int, col: int) -> bool:
        """检查角落死锁"""
        # 检查四个方向的墙壁情况
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右
        wall_count = 0
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if (not self._is_valid_position(new_row, new_col) or 
                self.board[new_row][new_col] == self.WALL):
                wall_count += 1
        
        # 如果被两面或更多面墙围住，且这些墙形成角落，则死锁
        if wall_count >= 2:
            # 检查是否真的形成角落（检查相邻的两面墙）
            wall_positions = []
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                if (not self._is_valid_position(new_row, new_col) or 
                    self.board[new_row][new_col] == self.WALL):
                    wall_positions.append((dr, dc))
            
            # 检查是否有相邻的墙壁形成角落
            for i, (dr1, dc1) in enumerate(wall_positions):
                for j, (dr2, dc2) in enumerate(wall_positions):
                    if i != j:
                        # 如果两个方向垂直，则形成角落
                        if (dr1 * dr2 == 0 and dc1 * dc2 == 0 and 
                            (dr1 != 0 or dc1 != 0) and (dr2 != 0 or dc2 != 0)):
                            return True
        
        return False
    
    def _is_edge_deadlock(self, row: int, col: int) -> bool:
        """检查边缘死锁（简化版）"""
        # 检查是否在靠墙的位置且无法推向目标
        # 这里使用简化的检查逻辑
        
        # 检查水平方向是否被阻挡
        left_blocked = (not self._is_valid_position(row, col - 1) or 
                       self.board[row][col - 1] == self.WALL or
                       (row, col - 1) in self.boxes)
        right_blocked = (not self._is_valid_position(row, col + 1) or 
                        self.board[row][col + 1] == self.WALL or
                        (row, col + 1) in self.boxes)
        
        # 检查垂直方向是否被阻挡
        up_blocked = (not self._is_valid_position(row - 1, col) or 
                     self.board[row - 1][col] == self.WALL or
                     (row - 1, col) in self.boxes)
        down_blocked = (not self._is_valid_position(row + 1, col) or 
                       self.board[row + 1][col] == self.WALL or
                       (row + 1, col) in self.boxes)
        
        # 如果水平和垂直方向都被完全阻挡，则可能死锁
        horizontal_blocked = left_blocked and right_blocked
        vertical_blocked = up_blocked and down_blocked
        
        return horizontal_blocked or vertical_blocked

    def get_winner(self) -> Optional[int]:
        """获取获胜者"""
        if not self.is_terminal():
            return None
        
        # 首先检查是否完成所有目标（正常胜利）
        if len(self.boxes_on_targets) == len(self.targets):
            if self.game_mode == 'cooperative':
                return 0  # 合作模式：共同获胜
            else:
                # 竞争模式：比较分数
                if self.player1_score > self.player2_score:
                    return 1
                elif self.player2_score > self.player1_score:
                    return 2
                else:
                    return 0  # 平局
        
        # 检查是否因为死锁而失败
        if self._check_deadlock():
            if self.game_mode == 'cooperative':
                return None  # 合作模式：共同失败
            else:
                # 竞争模式：当前回合的玩家失败，对手获胜
                if self.current_player == 1:
                    return 2  # 玩家1死锁，玩家2获胜
                else:
                    return 1  # 玩家2死锁，玩家1获胜
        
        # 检查是否超时或达到步数限制
        if self.is_timeout() or self.move_count >= self.max_steps:
            if self.game_mode == 'cooperative':
                return None  # 合作模式：共同失败
            else:
                # 竞争模式：比较当前分数
                if self.player1_score > self.player2_score:
                    return 1
                elif self.player2_score > self.player1_score:
                    return 2
                else:
                    return 0  # 平局
        
        return None
    
    def get_state(self) -> Dict[str, Any]:
        """获取当前游戏状态"""
        return {
            'board': [row[:] for row in self.board],  # 深拷贝
            'board_array': np.array(self._get_board_as_numbers()),
            'player1_pos': self.player1_pos,
            'player2_pos': self.player2_pos,
            'current_player': self.current_player,
            'player1_score': self.player1_score,
            'player2_score': self.player2_score,
            'player1_steps': self.player1_steps,
            'player2_steps': self.player2_steps,
            'boxes': list(self.boxes),
            'targets': list(self.targets),
            'boxes_on_targets': list(self.boxes_on_targets),
            'total_boxes': len(self.boxes),
            'completed_boxes': len(self.boxes_on_targets),
            'game_mode': self.game_mode,
            'level_id': self.level_id,
            'is_terminal': self.is_terminal(),
            'winner': self.get_winner(),
            'move_count': self.move_count
            # 注意：移除了valid_actions以避免递归调用
        }
    
    def _get_board_as_numbers(self) -> List[List[int]]:
        """将棋盘转换为数字表示"""
        # 符号到数字的映射
        symbol_map = {
            ' ': 0,      # 空地
            '#': 1,      # 墙壁
            '.': 2,      # 目标
            '$': 3,      # 箱子
            '*': 4,      # 箱子在目标上
            '@': 5,      # 玩家1
            '&': 6,      # 玩家2
            '+': 7,      # 玩家1在目标上
            '%': 8       # 玩家2在目标上
        }
        
        return [[symbol_map.get(cell, 0) for cell in row] for row in self.board]
    
    def render(self) -> str:
        """渲染游戏画面"""
        lines = []
        lines.append(f"=== 推箱子游戏 - 关卡 {self.level_id} ===")
        lines.append(f"模式: {'合作' if self.game_mode == 'cooperative' else '竞争'}")
        lines.append(f"当前玩家: {self.current_player}")
        lines.append(f"玩家1分数: {self.player1_score} (步数: {self.player1_steps})")
        lines.append(f"玩家2分数: {self.player2_score} (步数: {self.player2_steps})")
        lines.append(f"完成箱子: {len(self.boxes_on_targets)}/{len(self.targets)}")
        lines.append("")
        
        # 渲染棋盘
        for row in self.board:
            lines.append(''.join(row))
        
        lines.append("")
        lines.append("符号说明:")
        lines.append("@ = 玩家1, & = 玩家2, $ = 箱子, . = 目标, * = 箱子在目标上")
        lines.append("+ = 玩家1在目标上, % = 玩家2在目标上, # = 墙壁")
        
        return '\n'.join(lines)
    
    def clone(self) -> 'SokobanGame':
        """克隆游戏状态"""
        cloned = SokobanGame(self.level_id, self.game_mode)
        
        # 复制基本属性
        cloned.board = [row[:] for row in self.board]
        cloned.player1_pos = self.player1_pos
        cloned.player2_pos = self.player2_pos
        cloned.current_player = self.current_player
        cloned.player1_score = self.player1_score
        cloned.player2_score = self.player2_score
        cloned.player1_steps = self.player1_steps
        cloned.player2_steps = self.player2_steps
        cloned.move_count = self.move_count
        cloned.game_state = self.game_state
        
        # 复制集合
        cloned.boxes = self.boxes.copy()
        cloned.targets = self.targets.copy()
        cloned.boxes_on_targets = self.boxes_on_targets.copy()
        
        # 复制其他属性
        cloned.height = self.height
        cloned.width = self.width
        cloned.max_steps = self.max_steps
        cloned.time_limit = self.time_limit
        
        return cloned
    
    def get_action_space(self) -> List[str]:
        """获取动作空间"""
        return list(self.DIRECTIONS.keys())
    
    def get_observation_space(self) -> Tuple[int, int]:
        """获取观察空间"""
        return (self.height, self.width)
    
    def _fix_player_position_conflict(self):
        """修复玩家位置冲突"""
        if not self.player1_pos or not self.player2_pos:
            return
        
        if self.player1_pos != self.player2_pos:
            return
        
        # 为玩家2找一个新位置
        new_pos = None
        for row in range(self.height):
            for col in range(self.width):
                pos = (row, col)
                if (pos != self.player1_pos and 
                    (self.board[row][col] == self.FLOOR or 
                     self.board[row][col] == self.TARGET) and
                    pos not in self.boxes):
                    new_pos = pos
                    break
            if new_pos:
                break
        
        if new_pos:
            self.player2_pos = new_pos
        
        # 重新更新显示
        self._update_board_display()
    
    def _fix_initial_player_positions(self):
        """修复初始玩家位置冲突"""
        if not self.player1_pos:
            return
        
        # 为玩家2找一个新的位置
        original_p2_pos = self.player2_pos
        best_positions = []
        
        for row in range(self.height):
            for col in range(self.width):
                pos = (row, col)
                # 确保这个位置不是玩家1的位置，也不是墙壁或箱子
                if (pos != self.player1_pos and 
                    (self.board[row][col] == self.FLOOR or 
                     self.board[row][col] == self.TARGET) and
                    pos not in self.boxes):
                    # 计算与玩家1的距离，优先选择较远的位置
                    distance = abs(row - self.player1_pos[0]) + abs(col - self.player1_pos[1])
                    best_positions.append((distance, pos))
        
        if best_positions:
            # 按距离排序，选择距离最远的位置
            best_positions.sort(reverse=True)
            new_pos = best_positions[0][1]
            
            self.player2_pos = new_pos
            
            # 更新显示
            self._update_board_display()
