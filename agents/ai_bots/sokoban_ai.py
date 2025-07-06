"""
推箱子AI智能体
实现基于A*搜索的推箱子求解算法
"""

import heapq
import time
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import deque
import numpy as np
from agents.base_agent import BaseAgent


class SokobanAI(BaseAgent):
    """推箱子AI智能体 - 使用A*搜索算法"""
    
    def __init__(self, name: str = "Sokoban AI", player_id: int = 1, **kwargs):
        super().__init__(name, player_id)
        self.max_search_time = kwargs.get('max_search_time', 5.0)  # 最大搜索时间（秒）
        self.max_depth = kwargs.get('max_depth', 50)  # 最大搜索深度
        self.use_heuristic = kwargs.get('use_heuristic', True)
        
    def get_action(self, observation: Dict[str, Any], env) -> Optional[str]:
        """获取动作"""
        try:
            # 使用A*搜索找到最优动作
            action = self._search_best_action(observation, env)
            return action
        except Exception as e:
            print(f"AI thinking error: {e}")
            # 如果搜索失败，返回随机有效动作
            valid_actions = observation.get('valid_actions_mask')
            if hasattr(valid_actions, '__iter__') and any(valid_actions):
                actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
                for i, valid in enumerate(valid_actions):
                    if valid:
                        return actions[i]
            return 'UP'  # 默认动作
    
    def _search_best_action(self, observation: Dict[str, Any], env) -> Optional[str]:
        """使用A*搜索找到最佳动作"""
        start_time = time.time()
        
        # 获取当前状态
        current_state = self._observation_to_state(observation)
        
        # 如果已经完成，返回None
        if self._is_solved(current_state):
            return None
        
        # A*搜索
        frontier = []  # 优先队列
        heapq.heappush(frontier, (0, 0, current_state, []))  # (f_score, depth, state, path)
        
        visited = set()
        visited.add(self._state_to_key(current_state))
        
        best_action = None
        best_score = float('-inf')
        
        while frontier and time.time() - start_time < self.max_search_time:
            f_score, depth, state, path = heapq.heappop(frontier)
            
            # 检查深度限制
            if depth >= self.max_depth:
                continue
            
            # 尝试所有可能的动作
            for action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                new_state, success = self._simulate_action(state, action)
                
                if not success:
                    continue
                
                state_key = self._state_to_key(new_state)
                if state_key in visited:
                    continue
                
                visited.add(state_key)
                new_path = path + [action]
                
                # 评估新状态
                score = self._evaluate_state(new_state)
                
                # 如果找到解决方案
                if self._is_solved(new_state):
                    return new_path[0] if new_path else action
                
                # 更新最佳动作
                if len(new_path) == 1 and score > best_score:
                    best_score = score
                    best_action = action
                
                # 继续搜索
                if self.use_heuristic:
                    h_score = self._heuristic(new_state)
                    g_score = depth + 1
                    f_score = g_score + h_score
                else:
                    f_score = depth + 1
                
                heapq.heappush(frontier, (f_score, depth + 1, new_state, new_path))
        
        return best_action
    
    def _observation_to_state(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """将观察转换为状态"""
        board = observation['board']
        player_pos = None
        
        # 根据当前玩家确定位置
        if self.player_id == 1:
            player_pos = tuple(observation['player1_pos']) if observation['player1_pos'][0] >= 0 else None
        else:
            player_pos = tuple(observation['player2_pos']) if observation['player2_pos'][0] >= 0 else None
        
        # 提取箱子和目标位置
        boxes = set()
        targets = set()
        
        for row in range(board.shape[0]):
            for col in range(board.shape[1]):
                cell = board[row, col]
                if cell == 3 or cell == 4:  # 箱子或箱子在目标上
                    boxes.add((row, col))
                if cell == 2 or cell == 4 or cell == 7 or cell == 8:  # 目标点
                    targets.add((row, col))
        
        return {
            'board': board,
            'player_pos': player_pos,
            'boxes': boxes,
            'targets': targets
        }
    
    def _state_to_key(self, state: Dict[str, Any]) -> str:
        """将状态转换为唯一键"""
        player_pos = state['player_pos']
        boxes = tuple(sorted(state['boxes']))
        return f"{player_pos}_{boxes}"
    
    def _simulate_action(self, state: Dict[str, Any], action: str) -> Tuple[Dict[str, Any], bool]:
        """模拟执行动作"""
        board = state['board'].copy()
        player_pos = state['player_pos']
        boxes = state['boxes'].copy()
        targets = state['targets']
        
        if not player_pos:
            return state, False
        
        # 方向映射
        directions = {
            'UP': (-1, 0),
            'DOWN': (1, 0),
            'LEFT': (0, -1),
            'RIGHT': (0, 1)
        }
        
        if action not in directions:
            return state, False
        
        dr, dc = directions[action]
        new_row, new_col = player_pos[0] + dr, player_pos[1] + dc
        
        # 检查边界
        if (new_row < 0 or new_row >= board.shape[0] or 
            new_col < 0 or new_col >= board.shape[1]):
            return state, False
        
        # 检查墙壁
        if board[new_row, new_col] == 1:  # 墙壁
            return state, False
        
        # 检查是否有箱子
        if (new_row, new_col) in boxes:
            # 尝试推箱子
            box_new_row, box_new_col = new_row + dr, new_col + dc
            
            # 检查箱子新位置
            if (box_new_row < 0 or box_new_row >= board.shape[0] or 
                box_new_col < 0 or box_new_col >= board.shape[1]):
                return state, False
            
            if board[box_new_row, box_new_col] == 1:  # 墙壁
                return state, False
            
            if (box_new_row, box_new_col) in boxes:  # 另一个箱子
                return state, False
            
            # 移动箱子
            boxes.remove((new_row, new_col))
            boxes.add((box_new_row, box_new_col))
        
        # 创建新状态
        new_state = {
            'board': board,
            'player_pos': (new_row, new_col),
            'boxes': boxes,
            'targets': targets
        }
        
        return new_state, True
    
    def _is_solved(self, state: Dict[str, Any]) -> bool:
        """检查是否解决"""
        boxes = state['boxes']
        targets = state['targets']
        return boxes == targets
    
    def _evaluate_state(self, state: Dict[str, Any]) -> float:
        """评估状态质量"""
        boxes = state['boxes']
        targets = state['targets']
        player_pos = state['player_pos']
        
        # 基础分数：箱子在目标上的数量
        boxes_on_targets = len(boxes & targets)
        score = boxes_on_targets * 100
        
        # 惩罚：箱子距离目标的总距离
        total_box_distance = 0
        for box in boxes:
            if box not in targets:
                min_distance = float('inf')
                for target in targets:
                    if target not in boxes:  # 目标还没有箱子
                        distance = abs(box[0] - target[0]) + abs(box[1] - target[1])
                        min_distance = min(min_distance, distance)
                if min_distance != float('inf'):
                    total_box_distance += min_distance
        
        score -= total_box_distance * 2
        
        # 惩罚：玩家距离最近箱子的距离
        if player_pos and boxes:
            min_player_distance = float('inf')
            for box in boxes:
                if box not in targets:  # 优先考虑未完成的箱子
                    distance = abs(player_pos[0] - box[0]) + abs(player_pos[1] - box[1])
                    min_player_distance = min(min_player_distance, distance)
            
            if min_player_distance != float('inf'):
                score -= min_player_distance
        
        # 检测死锁：箱子在角落且不在目标上
        deadlock_penalty = self._detect_deadlocks(state)
        score -= deadlock_penalty * 1000
        
        return score
    
    def _heuristic(self, state: Dict[str, Any]) -> float:
        """启发式函数"""
        boxes = state['boxes']
        targets = state['targets']
        
        # 使用匈牙利算法的简化版本：最小距离匹配
        unmatched_boxes = [box for box in boxes if box not in targets]
        unmatched_targets = [target for target in targets if target not in boxes]
        
        if not unmatched_boxes:
            return 0
        
        total_distance = 0
        for box in unmatched_boxes:
            min_distance = float('inf')
            for target in unmatched_targets:
                distance = abs(box[0] - target[0]) + abs(box[1] - target[1])
                min_distance = min(min_distance, distance)
            
            if min_distance != float('inf'):
                total_distance += min_distance
        
        return total_distance
    
    def _detect_deadlocks(self, state: Dict[str, Any]) -> int:
        """检测死锁状态"""
        board = state['board']
        boxes = state['boxes']
        targets = state['targets']
        
        deadlock_count = 0
        
        for box in boxes:
            if box in targets:
                continue  # 已经在目标上，不是死锁
            
            row, col = box
            
            # 检查角落死锁
            if self._is_corner_deadlock(board, row, col):
                deadlock_count += 1
            
            # 检查边缘死锁
            if self._is_edge_deadlock(board, boxes, targets, row, col):
                deadlock_count += 1
        
        return deadlock_count
    
    def _is_corner_deadlock(self, board: np.ndarray, row: int, col: int) -> bool:
        """检查角落死锁"""
        # 检查四个角落情况
        corners = [
            ((-1, 0), (0, -1)),  # 左上
            ((-1, 0), (0, 1)),   # 右上
            ((1, 0), (0, -1)),   # 左下
            ((1, 0), (0, 1))     # 右下
        ]
        
        for (dr1, dc1), (dr2, dc2) in corners:
            r1, c1 = row + dr1, col + dc1
            r2, c2 = row + dr2, col + dc2
            
            # 检查边界
            if (0 <= r1 < board.shape[0] and 0 <= c1 < board.shape[1] and
                0 <= r2 < board.shape[0] and 0 <= c2 < board.shape[1]):
                
                # 如果两个相邻位置都是墙壁，形成角落死锁
                if board[r1, c1] == 1 and board[r2, c2] == 1:
                    return True
        
        return False
    
    def _is_edge_deadlock(self, board: np.ndarray, boxes: Set[Tuple[int, int]], 
                         targets: Set[Tuple[int, int]], row: int, col: int) -> bool:
        """检查边缘死锁（简化版本）"""
        # 检查是否靠墙且无法移动到目标
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        wall_count = 0
        
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if (r < 0 or r >= board.shape[0] or c < 0 or c >= board.shape[1] or
                board[r, c] == 1):
                wall_count += 1
        
        # 如果三面被墙围住，可能是死锁
        return wall_count >= 3


class SimpleSokobanAI(BaseAgent):
    """简单的推箱子AI - 使用贪心策略"""
    
    def __init__(self, name: str = "Simple Sokoban AI", player_id: int = 1):
        super().__init__(name, player_id)
    
    def get_action(self, observation: Dict[str, Any], env) -> Optional[str]:
        """获取动作 - 使用简单贪心策略"""
        try:
            # 获取有效动作
            valid_actions = self._get_valid_actions_from_mask(observation.get('valid_actions_mask'))
            
            if not valid_actions:
                return None
            
            # 获取当前状态
            board = observation['board']
            
            # 确定玩家位置
            if self.player_id == 1:
                player_pos = tuple(observation['player1_pos']) if observation['player1_pos'][0] >= 0 else None
            else:
                player_pos = tuple(observation['player2_pos']) if observation['player2_pos'][0] >= 0 else None
            
            if not player_pos:
                return valid_actions[0] if valid_actions else None
            
            # 简单策略：朝最近的未完成箱子移动
            best_action = self._find_best_action(board, player_pos, valid_actions)
            
            return best_action if best_action else valid_actions[0]
            
        except Exception as e:
            print(f"Simple AI error: {e}")
            # 返回第一个有效动作
            valid_actions = self._get_valid_actions_from_mask(observation.get('valid_actions_mask'))
            return valid_actions[0] if valid_actions else 'UP'
    
    def _get_valid_actions_from_mask(self, mask) -> List[str]:
        """从动作掩码获取有效动作"""
        if mask is None:
            return ['UP', 'DOWN', 'LEFT', 'RIGHT']
        
        actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        valid_actions = []
        
        try:
            # 安全地处理掩码
            if hasattr(mask, '__len__') and len(mask) >= 4:
                for i in range(min(4, len(mask))):
                    if mask[i]:  # 避免使用any()或all()
                        valid_actions.append(actions[i])
            else:
                valid_actions = actions  # 默认返回所有动作
        except:
            valid_actions = actions  # 出错时返回所有动作
        
        return valid_actions if valid_actions else actions
    
    def _find_best_action(self, board: np.ndarray, player_pos: Tuple[int, int], 
                         valid_actions: List[str]) -> Optional[str]:
        """找到最佳动作"""
        # 找到所有箱子和目标
        boxes = []
        targets = []
        
        for row in range(board.shape[0]):
            for col in range(board.shape[1]):
                cell = board[row, col]
                if cell == 3:  # 箱子
                    boxes.append((row, col))
                elif cell == 4:  # 箱子在目标上
                    boxes.append((row, col))
                
                if cell in [2, 4, 7, 8]:  # 目标点
                    targets.append((row, col))
        
        # 找到最近的未完成箱子
        incomplete_boxes = [box for box in boxes if box not in targets]
        
        if not incomplete_boxes:
            return valid_actions[0] if valid_actions else None
        
        # 找到最近的箱子
        min_distance = float('inf')
        target_box = None
        
        for box in incomplete_boxes:
            distance = abs(player_pos[0] - box[0]) + abs(player_pos[1] - box[1])
            if distance < min_distance:
                min_distance = distance
                target_box = box
        
        if not target_box:
            return valid_actions[0] if valid_actions else None
        
        # 朝目标箱子方向移动
        directions = {
            'UP': (-1, 0),
            'DOWN': (1, 0),
            'LEFT': (0, -1),
            'RIGHT': (0, 1)
        }
        
        best_action = None
        best_distance = float('inf')
        
        for action in valid_actions:
            if action in directions:
                dr, dc = directions[action]
                new_pos = (player_pos[0] + dr, player_pos[1] + dc)
                distance = abs(new_pos[0] - target_box[0]) + abs(new_pos[1] - target_box[1])
                
                if distance < best_distance:
                    best_distance = distance
                    best_action = action
        
        return best_action
