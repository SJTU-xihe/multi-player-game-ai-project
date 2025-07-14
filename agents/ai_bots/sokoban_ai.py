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
        self.max_search_time = kwargs.get('max_search_time', 5.0)  # 增加搜索时间
        self.max_depth = kwargs.get('max_depth', 50)  # 增加搜索深度
        self.use_heuristic = kwargs.get('use_heuristic', True)
        self.use_dynamic_depth = kwargs.get('use_dynamic_depth', True)
        self.cache_size = kwargs.get('cache_size', 50000)  # 增加缓存大小
        self.state_cache = {}  # 状态评估缓存
        self.deadlock_cache = set()  # 死锁状态缓存
        self.goal_push_cache = {}  # 目标推动路径缓存
        self.use_advanced_heuristic = kwargs.get('use_advanced_heuristic', True)  # 高级启发式
        self.prioritize_completion = kwargs.get('prioritize_completion', True)  # 优先完成策略
        
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
            if valid_actions is not None and hasattr(valid_actions, '__iter__'):
                actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
                for i, valid in enumerate(valid_actions):
                    if i < len(actions) and bool(valid):
                        return actions[i]
            return 'UP'  # 默认动作
    
    def _search_best_action(self, observation: Dict[str, Any], env) -> Optional[str]:
        """使用改进的A*搜索找到最佳动作"""
        start_time = time.time()
        
        # 获取当前状态
        current_state = self._observation_to_state(observation)
        
        # 如果已经完成，返回None
        if self._is_solved(current_state):
            return None
        
        # 优先检查一步解决方案
        immediate_solution = self._check_immediate_solution(current_state)
        if immediate_solution:
            return immediate_solution
        
        # 动态调整搜索深度
        if self.use_dynamic_depth:
            complexity = self._assess_state_complexity(current_state)
            max_depth = max(15, int(self.max_depth * (1.2 - complexity)))  # 更智能的深度调整
        else:
            max_depth = self.max_depth
        
        # 快速检查是否有明显的好动作
        quick_action = self._quick_action_check(current_state)
        if quick_action:
            return quick_action
        
        # 分层A*搜索：先搜索浅层，再搜索深层
        for depth_limit in [10, 20, max_depth]:
            action = self._layered_astar_search(current_state, depth_limit, start_time)
            if action or time.time() - start_time > self.max_search_time * 0.8:
                return action
        
        return None

    def _check_immediate_solution(self, state: Dict[str, Any]) -> Optional[str]:
        """检查是否有一步解决方案"""
        boxes = state['boxes']
        targets = state['targets']
        player_pos = state['player_pos']
        if not player_pos:
            return None
        incomplete_boxes = [box for box in boxes if box not in targets]
        if len(incomplete_boxes) != 1:
            return None
        box = incomplete_boxes[0]
        directions = {
            'UP': (-1, 0),
            'DOWN': (1, 0),
            'LEFT': (0, -1),
            'RIGHT': (0, 1)
        }
        for action, (dr, dc) in directions.items():
            if (player_pos[0] + dr, player_pos[1] + dc) == box:
                box_new_pos = (box[0] + dr, box[1] + dc)
                if box_new_pos in targets:
                    new_state, success = self._simulate_action(state, action)
                    if success:
                        return action
        return None

    def _layered_astar_search(self, initial_state: Dict[str, Any], max_depth: int, start_time: float) -> Optional[str]:
        """分层A*搜索"""
        frontier = []
        heapq.heappush(frontier, (0, 0, initial_state, []))
        visited = set()
        visited.add(self._state_to_key(initial_state))
        best_action = None
        best_score = float('-inf')
        nodes_explored = 0
        max_nodes = min(3000, max_depth * 150)
        while frontier and time.time() - start_time < self.max_search_time and nodes_explored < max_nodes:
            f_score, depth, state, path = heapq.heappop(frontier)
            nodes_explored += 1
            if depth >= max_depth:
                continue
            state_key = self._state_to_key(state)
            if state_key in self.deadlock_cache:
                continue
            for action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                new_state, success = self._simulate_action(state, action)
                if not success:
                    continue
                new_state_key = self._state_to_key(new_state)
                if new_state_key in visited:
                    continue
                visited.add(new_state_key)
                new_path = path + [action]
                if self._advanced_deadlock_check(new_state):
                    self.deadlock_cache.add(new_state_key)
                    continue
                if self._is_solved(new_state):
                    return new_path[0] if new_path else action
                if self.use_advanced_heuristic:
                    score = self._evaluate_state_advanced(new_state)
                else:
                    score = self._evaluate_state_cached(new_state)
                if len(new_path) == 1 and score > best_score:
                    best_score = score
                    best_action = action
                if score < -2000:
                    continue
                if self.use_advanced_heuristic:
                    h_score = self._advanced_heuristic(new_state)
                else:
                    h_score = self._heuristic_cached(new_state)
                g_score = depth + 1
                f_score = g_score + h_score
                heapq.heappush(frontier, (f_score, depth + 1, new_state, new_path))
        return best_action

    def _evaluate_state_advanced(self, state: Dict[str, Any]) -> float:
        """高级状态评估函数"""
        boxes = state['boxes']
        targets = state['targets']
        player_pos = state['player_pos']
        if self._is_solved(state):
            return 100000
        completed_boxes = len(boxes & targets)
        total_targets = len(targets)
        base_score = completed_boxes * 5000
        progressive_bonus = completed_boxes * completed_boxes * 1000
        incomplete_boxes = [box for box in boxes if box not in targets]
        available_targets = [target for target in targets if target not in boxes]
        distance_penalty = 0
        if incomplete_boxes and available_targets:
            min_total_distance = self._calculate_optimal_assignment(incomplete_boxes, available_targets)
            distance_penalty = min_total_distance * 50
        push_bonus = 0
        if player_pos and incomplete_boxes:
            push_bonus = self._calculate_push_potential(player_pos, incomplete_boxes, available_targets, state)
        deadlock_penalty = 0
        for box in incomplete_boxes:
            if self._is_box_deadlocked(box, state):
                deadlock_penalty += 10000
        total_score = (base_score + progressive_bonus + push_bonus - distance_penalty - deadlock_penalty)
        return total_score

    def _advanced_heuristic(self, state: Dict[str, Any]) -> float:
        boxes = state['boxes']
        targets = state['targets']
        incomplete_boxes = [box for box in boxes if box not in targets]
        available_targets = [target for target in targets if target not in boxes]
        if not incomplete_boxes:
            return 0
        total_distance = self._calculate_optimal_assignment(incomplete_boxes, available_targets)
        push_steps_estimate = len(incomplete_boxes) * 2
        return total_distance + push_steps_estimate

    def _calculate_optimal_assignment(self, boxes: List[Tuple[int, int]], targets: List[Tuple[int, int]]) -> float:
        if not boxes or not targets:
            return 0
        total_distance = 0
        used_targets = set()
        for box in boxes:
            min_distance = float('inf')
            best_target = None
            for target in targets:
                if target not in used_targets:
                    distance = abs(box[0] - target[0]) + abs(box[1] - target[1])
                    if distance < min_distance:
                        min_distance = distance
                        best_target = target
            if best_target:
                used_targets.add(best_target)
                total_distance += min_distance
        return total_distance

    def _calculate_push_potential(self, player_pos: Tuple[int, int], incomplete_boxes: List[Tuple[int, int]], available_targets: List[Tuple[int, int]], state: Dict[str, Any]) -> float:
        bonus = 0
        for box in incomplete_boxes:
            player_to_box = abs(player_pos[0] - box[0]) + abs(player_pos[1] - box[1])
            if player_to_box <= 2:
                directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                for dr, dc in directions:
                    push_from = (box[0] - dr, box[1] - dc)
                    push_to = (box[0] + dr, box[1] + dc)
                    if (push_from == player_pos and self._is_valid_push_position(push_to, state)):
                        if available_targets:
                            min_target_distance = min(
                                abs(push_to[0] - target[0]) + abs(push_to[1] - target[1])
                                for target in available_targets
                            )
                            current_min_distance = min(
                                abs(box[0] - target[0]) + abs(box[1] - target[1])
                                for target in available_targets
                            )
                            if min_target_distance < current_min_distance:
                                improvement = current_min_distance - min_target_distance
                                bonus += improvement * 100
                            if push_to in available_targets:
                                bonus += 2000
        return bonus

    def _is_valid_push_position(self, pos: Tuple[int, int], state: Dict[str, Any]) -> bool:
        board = state['board']
        boxes = state['boxes']
        row, col = pos
        if row < 0 or row >= board.shape[0] or col < 0 or col >= board.shape[1]:
            return False
        if board[row, col] == 1:
            return False
        if pos in boxes:
            return False
        return True

    def _advanced_deadlock_check(self, state: Dict[str, Any]) -> bool:
        board = state['board']
        boxes = state['boxes']
        targets = state['targets']
        for box in boxes:
            if box not in targets:
                if self._is_box_deadlocked(box, state):
                    return True
        return False

    def _is_box_deadlocked(self, box: Tuple[int, int], state: Dict[str, Any]) -> bool:
        board = state['board']
        boxes = state['boxes']
        targets = state['targets']
        if box in targets:
            return False
        row, col = box
        if self._is_corner_deadlock(board, row, col):
            return True
        wall_count = 0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if (new_row < 0 or new_row >= board.shape[0] or new_col < 0 or new_col >= board.shape[1] or board[new_row, new_col] == 1):
                wall_count += 1
            elif (new_row, new_col) in boxes:
                wall_count += 0.5
        return wall_count >= 3

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
        """评估状态质量 - 激进的推箱子完成意识"""
        boxes = state['boxes']
        targets = state['targets']
        player_pos = state['player_pos']
        
        # 如果已经完成，给极高奖励
        if self._is_solved(state):
            return 10000
        
        # 基础分数：箱子在目标上的数量（最重要）
        boxes_on_targets = len(boxes & targets)
        score = boxes_on_targets * 1000  # 每个完成的箱子1000分
        
        # 简化距离计算
        unmatched_boxes = [box for box in boxes if box not in targets]
        unmatched_targets = [target for target in targets if target not in boxes]
        
        # 最近距离奖励：每个箱子到最近目标的距离
        for box in unmatched_boxes:
            min_distance = float('inf')
            for target in unmatched_targets:
                distance = abs(box[0] - target[0]) + abs(box[1] - target[1])
                min_distance = min(min_distance, distance)
            
            if min_distance != float('inf'):
                # 距离越近，分数越高
                score += max(0, 100 - min_distance * 10)
        
        # 推箱子激励：如果玩家靠近可推动的箱子，给额外奖励
        if player_pos:
            for box in unmatched_boxes:
                # 玩家到箱子的距离
                player_to_box = abs(player_pos[0] - box[0]) + abs(player_pos[1] - box[1])
                
                # 如果玩家紧挨着箱子，给高分奖励
                if player_to_box == 1:
                    # 检查是否可以推向目标方向
                    box_row, box_col = box
                    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                    
                    for dr, dc in directions:
                        new_box_pos = (box_row + dr, box_col + dc)
                        player_push_pos = (box_row - dr, box_col - dc)
                        
                        # 检查推动的有效性
                        if (0 <= new_box_pos[0] < state['board'].shape[0] and
                            0 <= new_box_pos[1] < state['board'].shape[1] and
                            state['board'][new_box_pos[0], new_box_pos[1]] != 1 and
                            new_box_pos not in boxes and
                            player_push_pos == player_pos):
                            
                            # 计算推动后箱子到最近目标的距离
                            min_target_distance = float('inf')
                            for target in unmatched_targets:
                                target_distance = abs(new_box_pos[0] - target[0]) + abs(new_box_pos[1] - target[1])
                                min_target_distance = min(min_target_distance, target_distance)
                            
                            # 如果推动让箱子更靠近目标，给高奖励
                            current_distance = float('inf')
                            for target in unmatched_targets:
                                current_distance = min(current_distance, 
                                    abs(box[0] - target[0]) + abs(box[1] - target[1]))
                            
                            if min_target_distance < current_distance:
                                score += 200  # 好的推动奖励
                            elif new_box_pos in targets:
                                score += 500  # 直接推到目标超高奖励
        
        # 死锁检测（简化版）
        deadlock_count = 0
        for box in unmatched_boxes:
            row, col = box
            # 检查是否在角落且不是目标
            corner_walls = 0
            if (row > 0 and state['board'][row-1, col] == 1) or row == 0:
                corner_walls += 1
            if (row < state['board'].shape[0]-1 and state['board'][row+1, col] == 1) or row == state['board'].shape[0]-1:
                corner_walls += 1
            if (col > 0 and state['board'][row, col-1] == 1) or col == 0:
                corner_walls += 1
            if (col < state['board'].shape[1]-1 and state['board'][row, col+1] == 1) or col == state['board'].shape[1]-1:
                corner_walls += 1
            
            if corner_walls >= 2:  # 在角落
                deadlock_count += 1
        
        score -= deadlock_count * 1000  # 死锁严重惩罚
        
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

    def _assess_state_complexity(self, state: Dict[str, Any]) -> float:
        """评估状态复杂度（0-1）"""
        board = state['board']
        boxes = state['boxes']
        targets = state['targets']
        
        # 基础复杂度因子
        total_cells = board.shape[0] * board.shape[1]
        boxes_count = len(boxes)
        
        # 未完成箱子数量
        unfinished_boxes = len([box for box in boxes if box not in targets])
        
        # 箱子分散度
        if len(boxes) > 1:
            box_positions = list(boxes)
            scatter = 0
            for i in range(len(box_positions)):
                for j in range(i + 1, len(box_positions)):
                    scatter += abs(box_positions[i][0] - box_positions[j][0]) + abs(box_positions[i][1] - box_positions[j][1])
            scatter /= (len(box_positions) * (len(box_positions) - 1) / 2)
        else:
            scatter = 0
        
        # 归一化复杂度
        complexity = min(1.0, (unfinished_boxes * 0.3 + boxes_count * 0.2 + scatter * 0.01))
        
        return complexity
    
    def _quick_action_check(self, state: Dict[str, Any]) -> Optional[str]:
        """快速检查是否有明显的好动作 - 优化推箱子意识"""
        boxes = state['boxes']
        targets = state['targets']
        player_pos = state['player_pos']
        
        if not player_pos:
            return None
        
        directions = {
            'UP': (-1, 0),
            'DOWN': (1, 0),
            'LEFT': (0, -1),
            'RIGHT': (0, 1)
        }
        
        # 优先级1: 检查是否可以直接完成一个箱子
        for action, (dr, dc) in directions.items():
            new_row, new_col = player_pos[0] + dr, player_pos[1] + dc
            
            # 检查是否有箱子
            if (new_row, new_col) in boxes:
                box_new_row, box_new_col = new_row + dr, new_col + dc
                
                # 检查推箱子后是否在目标上
                if (box_new_row, box_new_col) in targets:
                    # 验证这个动作是否合法
                    new_state, success = self._simulate_action(state, action)
                    if success:
                        return action
        
        # 优先级2: 检查是否可以将箱子推向目标方向
        best_action = None
        best_improvement = 0
        
        for action, (dr, dc) in directions.items():
            new_row, new_col = player_pos[0] + dr, player_pos[1] + dc
            
            # 检查是否有箱子可推
            if (new_row, new_col) in boxes and (new_row, new_col) not in targets:
                box_new_row, box_new_col = new_row + dr, new_col + dc
                
                # 验证推箱子是否合法
                new_state, success = self._simulate_action(state, action)
                if not success:
                    continue
                
                # 计算推箱子后的改善程度
                old_min_dist = float('inf')
                new_min_dist = float('inf')
                
                # 计算原位置到目标的最小距离
                for target in targets:
                    if target not in boxes:
                        old_dist = abs(new_row - target[0]) + abs(new_col - target[1])
                        old_min_dist = min(old_min_dist, old_dist)
                        
                        new_dist = abs(box_new_row - target[0]) + abs(box_new_col - target[1])
                        new_min_dist = min(new_min_dist, new_dist)
                
                # 如果推箱子让箱子更接近目标
                if new_min_dist < old_min_dist:
                    improvement = old_min_dist - new_min_dist
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_action = action
        
        if best_action and best_improvement > 0:
            return best_action
        
        # 优先级3: 检查是否可以向有用的箱子移动
        unmatched_boxes = [box for box in boxes if box not in targets]
        if unmatched_boxes:
            best_action = None
            best_distance = float('inf')
            
            for action, (dr, dc) in directions.items():
                new_pos = (player_pos[0] + dr, player_pos[1] + dc)
                
                # 验证移动是否合法
                new_state, success = self._simulate_action(state, action)
                if not success:
                    continue
                
                # 计算到最近未完成箱子的距离
                min_distance = float('inf')
                for box in unmatched_boxes:
                    distance = abs(new_pos[0] - box[0]) + abs(new_pos[1] - box[1])
                    min_distance = min(min_distance, distance)
                
                if min_distance < best_distance:
                    best_distance = min_distance
                    best_action = action
            
            return best_action
        
        return None
    
    def _quick_deadlock_check(self, state: Dict[str, Any]) -> bool:
        """快速死锁检测"""
        board = state['board']
        boxes = state['boxes']
        targets = state['targets']
        
        # 检查简单的角落死锁
        for box in boxes:
            if box not in targets:
                row, col = box
                if self._is_corner_deadlock(board, row, col):
                    return True
        
        return False
    
    def _evaluate_state_cached(self, state: Dict[str, Any]) -> float:
        """带缓存的状态评估"""
        state_key = self._state_to_key(state)
        
        if state_key in self.state_cache:
            return self.state_cache[state_key]
        
        score = self._evaluate_state(state)
        self.state_cache[state_key] = score
        
        return score
    
    def _heuristic_cached(self, state: Dict[str, Any]) -> float:
        """带缓存的启发式函数"""
        state_key = f"h_{self._state_to_key(state)}"
        
        if state_key in self.state_cache:
            return self.state_cache[state_key]
        
        score = self._heuristic(state)
        self.state_cache[state_key] = score
        
        return score


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
                    if bool(mask[i]):  # 明确转换为布尔值
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
