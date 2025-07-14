"""
通用搜索算法AI智能体
支持BFS、A*、DFS等多种搜索算法
适用于多种游戏环境
"""

import random
import heapq
import numpy as np
from typing import List, Tuple, Optional, Dict, Set, Any
from collections import deque
from agents.base_agent import BaseAgent


class SearchAI(BaseAgent):
    """通用搜索算法AI智能体"""
    
    def __init__(self, name="SearchAI", player_id=1, search_algorithm="bfs", max_depth=10):
        """
        初始化搜索AI
        
        Args:
            name: AI名称
            player_id: 玩家ID
            search_algorithm: 搜索算法类型 ("bfs", "dfs", "astar", "dijkstra")
            max_depth: 最大搜索深度
        """
        super().__init__(name, player_id)
        self.search_algorithm = search_algorithm.lower()
        self.max_depth = max_depth
        self.visited_positions = set()
        self.path_cache = {}  # 路径缓存
        
    def get_action(self, observation, env):
        """获取动作"""
        try:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                return None
            
            if len(valid_actions) == 1:
                return valid_actions[0]
            
            # 根据游戏类型选择不同的搜索策略
            game_type = self._detect_game_type(env)
            
            if game_type == "snake":
                return self._snake_search_action(observation, env, valid_actions)
            elif game_type == "sokoban":
                return self._sokoban_search_action(observation, env, valid_actions)
            elif game_type == "gomoku":
                return self._gomoku_search_action(observation, env, valid_actions)
            else:
                # 通用搜索策略
                return self._generic_search_action(observation, env, valid_actions)
                
        except Exception as e:
            print(f"SearchAI error: {e}")
            return random.choice(env.get_valid_actions() or [0])
    
    def _detect_game_type(self, env) -> str:
        """检测游戏类型"""
        class_name = env.__class__.__name__.lower()
        if "snake" in class_name:
            return "snake"
        elif "sokoban" in class_name:
            return "sokoban"
        elif "gomoku" in class_name:
            return "gomoku"
        else:
            return "generic"
    
    def _snake_search_action(self, observation, env, valid_actions):
        """贪吃蛇专用搜索策略"""
        game = env.game
        
        # 获取当前蛇的信息
        if self.player_id == 1:
            my_snake = game.snake1
            enemy_snake = game.snake2
        else:
            my_snake = game.snake2
            enemy_snake = game.snake1
        
        if not my_snake:
            return random.choice(valid_actions)
        
        my_head = my_snake[0]
        
        # 获取食物位置（取最近的食物）
        if hasattr(game, 'foods') and game.foods:
            food_pos = min(game.foods, key=lambda f: abs(f[0] - my_head[0]) + abs(f[1] - my_head[1]))
        else:
            # 如果没有食物，随机选择一个目标位置
            food_pos = (game.board_size // 2, game.board_size // 2)
        
        # 使用选定的搜索算法寻找到食物的最优路径
        if self.search_algorithm == "astar":
            path = self._astar_search_snake(my_head, food_pos, my_snake, enemy_snake, env)
        elif self.search_algorithm == "bfs":
            path = self._bfs_search_snake(my_head, food_pos, my_snake, enemy_snake, env)
        elif self.search_algorithm == "dfs":
            path = self._dfs_search_snake(my_head, food_pos, my_snake, enemy_snake, env)
        else:
            path = self._bfs_search_snake(my_head, food_pos, my_snake, enemy_snake, env)
        
        if path and len(path) > 1:
            next_pos = path[1]
            action = self._position_to_action(my_head, next_pos)
            if action in valid_actions:
                return action
        
        # 如果没有找到路径，选择安全的动作
        return self._safe_action_snake(my_head, my_snake, enemy_snake, env, valid_actions)
    
    def _sokoban_search_action(self, observation, env, valid_actions):
        """推箱子专用搜索策略 - 大幅增强"""
        game = env.game
        
        # 获取玩家位置
        if hasattr(game, 'players') and self.player_id <= len(game.players):
            player_pos = game.players[self.player_id - 1]
        else:
            # 从观察中找到玩家位置
            player_pos = self._find_player_position(observation, self.player_id)
        
        if not player_pos:
            return random.choice(valid_actions)
        
        # 寻找所有箱子和目标
        boxes = self._find_sokoban_boxes(observation)
        targets = self._find_sokoban_targets(observation)
        
        if not boxes or not targets:
            return random.choice(valid_actions)
        
        # 使用多层搜索策略
        best_action = self._multi_layer_sokoban_search(
            player_pos, boxes, targets, observation, env, valid_actions
        )
        
        return best_action if best_action else random.choice(valid_actions)
    
    def _multi_layer_sokoban_search(self, player_pos, boxes, targets, observation, env, valid_actions):
        """多层推箱子搜索算法"""
        
        # 第1层：直接完成检查
        immediate_action = self._check_immediate_completion(player_pos, boxes, targets, observation)
        if immediate_action and immediate_action in valid_actions:
            return immediate_action
        
        # 第2层：最优推箱子策略
        push_action = self._optimal_push_strategy(player_pos, boxes, targets, observation, valid_actions)
        if push_action:
            return push_action
        
        # 第3层：战略定位搜索
        positioning_action = self._strategic_positioning_search(player_pos, boxes, targets, observation, valid_actions)
        if positioning_action:
            return positioning_action
        
        # 第4层：A*路径搜索到最有价值的位置
        pathfinding_action = self._astar_to_valuable_position(player_pos, boxes, targets, observation, valid_actions)
        if pathfinding_action:
            return pathfinding_action
        
        return None
    
    def _check_immediate_completion(self, player_pos, boxes, targets, observation):
        """检查是否可以立即完成一个箱子"""
        directions = {
            'UP': (-1, 0),
            'DOWN': (1, 0),
            'LEFT': (0, -1),
            'RIGHT': (0, 1)
        }
        
        for action, (dr, dc) in directions.items():
            adjacent_pos = (player_pos[0] + dr, player_pos[1] + dc)
            
            # 如果相邻位置有箱子
            if adjacent_pos in boxes:
                box_new_pos = (adjacent_pos[0] + dr, adjacent_pos[1] + dc)
                
                # 检查推动后是否在目标上
                if (box_new_pos in targets and 
                    self._is_valid_sokoban_position(box_new_pos, observation, boxes)):
                    return action
        
        return None
    
    def _optimal_push_strategy(self, player_pos, boxes, targets, observation, valid_actions):
        """最优推箱子策略"""
        directions = {
            'UP': (-1, 0),
            'DOWN': (1, 0),
            'LEFT': (0, -1),
            'RIGHT': (0, 1)
        }
        
        best_action = None
        best_improvement = 0
        
        for action in valid_actions:
            if action not in directions:
                continue
                
            dr, dc = directions[action]
            adjacent_pos = (player_pos[0] + dr, player_pos[1] + dc)
            
            # 如果可以推箱子
            if adjacent_pos in boxes:
                box_new_pos = (adjacent_pos[0] + dr, adjacent_pos[1] + dc)
                
                if self._is_valid_sokoban_position(box_new_pos, observation, boxes):
                    # 计算推动的价值改善
                    improvement = self._calculate_push_improvement(
                        adjacent_pos, box_new_pos, boxes, targets
                    )
                    
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_action = action
        
        return best_action if best_improvement > 0 else None
    
    def _strategic_positioning_search(self, player_pos, boxes, targets, observation, valid_actions):
        """战略定位搜索 - 找到可以推动箱子的最佳位置"""
        directions = {
            'UP': (-1, 0),
            'DOWN': (1, 0),
            'LEFT': (0, -1),
            'RIGHT': (0, 1)
        }
        
        # 找到所有可推动的箱子位置
        valuable_positions = []
        incomplete_boxes = [box for box in boxes if box not in targets]
        
        for box in incomplete_boxes:
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                push_from_pos = (box[0] - dr, box[1] - dc)  # 玩家推动位置
                push_to_pos = (box[0] + dr, box[1] + dc)    # 箱子目标位置
                
                # 检查推动的有效性和价值
                if (self._is_valid_sokoban_position(push_from_pos, observation, boxes) and
                    self._is_valid_sokoban_position(push_to_pos, observation, boxes)):
                    
                    value = self._evaluate_push_position_value(box, push_to_pos, targets, boxes)
                    if value > 0:
                        valuable_positions.append((push_from_pos, value))
        
        if not valuable_positions:
            return None
        
        # 选择最有价值且最近的位置
        best_position = max(valuable_positions, key=lambda x: x[1] - abs(x[0][0] - player_pos[0]) - abs(x[0][1] - player_pos[1]))
        target_position = best_position[0]
        
        # 朝目标位置移动
        best_action = None
        min_distance = float('inf')
        
        for action in valid_actions:
            if action in directions:
                dr, dc = directions[action]
                new_pos = (player_pos[0] + dr, player_pos[1] + dc)
                distance = abs(new_pos[0] - target_position[0]) + abs(new_pos[1] - target_position[1])
                
                if distance < min_distance:
                    min_distance = distance
                    best_action = action
        
        return best_action
    
    def _astar_to_valuable_position(self, player_pos, boxes, targets, observation, valid_actions):
        """A*搜索到最有价值的位置"""
        # 简化版A*搜索，找到最近的有价值箱子
        incomplete_boxes = [box for box in boxes if box not in targets]
        if not incomplete_boxes:
            return None
        
        # 找到最有价值的箱子（最接近目标的箱子）
        most_valuable_box = min(incomplete_boxes, 
                               key=lambda box: min(abs(box[0] - t[0]) + abs(box[1] - t[1]) for t in targets))
        
        # 朝最有价值的箱子移动
        directions = {
            'UP': (-1, 0),
            'DOWN': (1, 0),
            'LEFT': (0, -1),
            'RIGHT': (0, 1)
        }
        
        best_action = None
        min_distance = float('inf')
        
        for action in valid_actions:
            if action in directions:
                dr, dc = directions[action]
                new_pos = (player_pos[0] + dr, player_pos[1] + dc)
                distance = abs(new_pos[0] - most_valuable_box[0]) + abs(new_pos[1] - most_valuable_box[1])
                
                if distance < min_distance:
                    min_distance = distance
                    best_action = action
        
        return best_action
    
    def _calculate_push_improvement(self, old_box_pos, new_box_pos, boxes, targets):
        """计算推动箱子的改善程度"""
        available_targets = [t for t in targets if t not in boxes]
        if not available_targets:
            return 0
        
        # 计算推动前后到最近目标的距离
        old_min_dist = min(abs(old_box_pos[0] - t[0]) + abs(old_box_pos[1] - t[1]) for t in available_targets)
        new_min_dist = min(abs(new_box_pos[0] - t[0]) + abs(new_box_pos[1] - t[1]) for t in available_targets)
        
        # 如果直接推到目标
        if new_box_pos in targets:
            return 1000
        
        # 返回距离改善
        return max(0, old_min_dist - new_min_dist)
    
    def _evaluate_push_position_value(self, box_pos, push_to_pos, targets, boxes):
        """评估推动位置的价值"""
        available_targets = [t for t in targets if t not in boxes]
        if not available_targets:
            return 0
        
        # 如果直接推到目标
        if push_to_pos in targets:
            return 1000
        
        # 计算推动后到最近目标的距离改善
        current_min_dist = min(abs(box_pos[0] - t[0]) + abs(box_pos[1] - t[1]) for t in available_targets)
        new_min_dist = min(abs(push_to_pos[0] - t[0]) + abs(push_to_pos[1] - t[1]) for t in available_targets)
        
        improvement = current_min_dist - new_min_dist
        return max(0, improvement * 10)  # 放大改善价值
    
    def _find_sokoban_boxes(self, observation):
        """找到推箱子中的所有箱子"""
        boxes = []
        if isinstance(observation, dict) and 'board' in observation:
            board = observation['board']
            for row in range(board.shape[0]):
                for col in range(board.shape[1]):
                    if board[row, col] in [3, 4]:  # 箱子或箱子在目标上
                        boxes.append((row, col))
        elif isinstance(observation, np.ndarray):
            positions = np.where((observation == 3) | (observation == 4))
            boxes = [(positions[0][i], positions[1][i]) for i in range(len(positions[0]))]
        return boxes
    
    def _find_sokoban_targets(self, observation):
        """找到推箱子中的所有目标"""
        targets = []
        if isinstance(observation, dict) and 'board' in observation:
            board = observation['board']
            for row in range(board.shape[0]):
                for col in range(board.shape[1]):
                    if board[row, col] in [2, 4, 7, 8]:  # 目标点
                        targets.append((row, col))
        elif isinstance(observation, np.ndarray):
            positions = np.where((observation == 2) | (observation == 4) | (observation == 7) | (observation == 8))
            targets = [(positions[0][i], positions[1][i]) for i in range(len(positions[0]))]
        return targets
    
    def _is_valid_sokoban_position(self, pos, observation, boxes):
        """检查推箱子位置是否有效"""
        row, col = pos
        
        # 获取棋盘
        if isinstance(observation, dict) and 'board' in observation:
            board = observation['board']
        else:
            board = observation
        
        # 检查边界
        if row < 0 or row >= board.shape[0] or col < 0 or col >= board.shape[1]:
            return False
        
        # 检查是否是墙壁
        if board[row, col] == 1:
            return False
        
        # 检查是否有其他箱子
        if pos in boxes:
            return False
        
        return True
    
    def _gomoku_search_action(self, observation, env, valid_actions):
        """五子棋专用搜索策略"""
        # 使用minimax或者简单的启发式搜索
        best_action = self._search_gomoku_move(observation, env, valid_actions)
        return best_action if best_action else random.choice(valid_actions)
    
    def _generic_search_action(self, observation, env, valid_actions):
        """通用搜索策略"""
        # 简单的随机选择或基于观察的启发式
        return random.choice(valid_actions)
    
    def _bfs_search_snake(self, start, goal, my_snake, enemy_snake, env):
        """BFS搜索（贪吃蛇）"""
        queue = deque([(start, [start])])
        visited = {start}
        board_size = env.game.board_size
        
        while queue:
            current, path = queue.popleft()
            
            if len(path) > self.max_depth:
                continue
            
            if current == goal:
                return path
            
            # 探索四个方向
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_pos = (current[0] + dx, current[1] + dy)
                
                if (next_pos not in visited and 
                    self._is_valid_position_snake(next_pos, my_snake, enemy_snake, board_size)):
                    visited.add(next_pos)
                    queue.append((next_pos, path + [next_pos]))
        
        return []
    
    def _dfs_search_snake(self, start, goal, my_snake, enemy_snake, env):
        """DFS搜索（贪吃蛇）"""
        def dfs_recursive(current, path, visited, depth):
            if depth > self.max_depth:
                return None
            
            if current == goal:
                return path
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_pos = (current[0] + dx, current[1] + dy)
                
                if (next_pos not in visited and 
                    self._is_valid_position_snake(next_pos, my_snake, enemy_snake, env.game.board_size)):
                    visited.add(next_pos)
                    result = dfs_recursive(next_pos, path + [next_pos], visited, depth + 1)
                    if result:
                        return result
                    visited.remove(next_pos)
            
            return None
        
        return dfs_recursive(start, [start], {start}, 0) or []
    
    def _astar_search_snake(self, start, goal, my_snake, enemy_snake, env):
        """A*搜索（贪吃蛇）"""
        def heuristic(pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        
        heap = [(0, 0, start, [start])]  # (f_score, g_score, position, path)
        visited = {start: 0}
        board_size = env.game.board_size
        
        while heap:
            f_score, g_score, current, path = heapq.heappop(heap)
            
            if g_score > self.max_depth:
                continue
            
            if current == goal:
                return path
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_pos = (current[0] + dx, current[1] + dy)
                new_g_score = g_score + 1
                
                if (self._is_valid_position_snake(next_pos, my_snake, enemy_snake, board_size) and
                    (next_pos not in visited or new_g_score < visited[next_pos])):
                    visited[next_pos] = new_g_score
                    h_score = heuristic(next_pos, goal)
                    f_score = new_g_score + h_score
                    heapq.heappush(heap, (f_score, new_g_score, next_pos, path + [next_pos]))
        
        return []
    
    def _is_valid_position_snake(self, pos, my_snake, enemy_snake, board_size):
        """检查位置是否有效（贪吃蛇）"""
        x, y = pos
        
        # 检查边界
        if x < 0 or x >= board_size or y < 0 or y >= board_size:
            return False
        
        # 检查是否与自己的蛇身碰撞
        if pos in my_snake:
            return False
        
        # 检查是否与对手的蛇身碰撞
        if enemy_snake and pos in enemy_snake:
            return False
        
        return True
    
    def _safe_action_snake(self, head, my_snake, enemy_snake, env, valid_actions):
        """选择安全动作（贪吃蛇）"""
        safe_actions = []
        board_size = env.game.board_size
        
        action_map = {
            0: (0, -1),   # UP
            1: (0, 1),    # DOWN
            2: (-1, 0),   # LEFT
            3: (1, 0)     # RIGHT
        }
        
        for action in valid_actions:
            if action in action_map:
                dx, dy = action_map[action]
                next_pos = (head[0] + dx, head[1] + dy)
                
                if self._is_valid_position_snake(next_pos, my_snake, enemy_snake, board_size):
                    safe_actions.append(action)
        
        return random.choice(safe_actions) if safe_actions else random.choice(valid_actions)
    
    def _position_to_action(self, current_pos, next_pos):
        """将位置转换为动作"""
        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1]
        
        if dx == 0 and dy == -1:
            return 0  # UP
        elif dx == 0 and dy == 1:
            return 1  # DOWN
        elif dx == -1 and dy == 0:
            return 2  # LEFT
        elif dx == 1 and dy == 0:
            return 3  # RIGHT
        else:
            return 0  # 默认向上
    
    def _find_player_position(self, observation, player_id):
        """从观察中找到玩家位置"""
        # 根据观察格式查找玩家位置
        if isinstance(observation, np.ndarray):
            positions = np.where(observation == player_id + 3)  # 假设玩家标记为4, 5等
            if len(positions[0]) > 0:
                return (positions[0][0], positions[1][0])
        return None
    
    def _find_boxes(self, observation):
        """找到所有箱子位置"""
        if isinstance(observation, np.ndarray):
            positions = np.where(observation == 2)  # 假设箱子标记为2
            return [(positions[0][i], positions[1][i]) for i in range(len(positions[0]))]
        return []
    
    def _find_targets(self, observation):
        """找到所有目标位置"""
        if isinstance(observation, np.ndarray):
            positions = np.where(observation == 3)  # 假设目标标记为3
            return [(positions[0][i], positions[1][i]) for i in range(len(positions[0]))]
        return []
    
    def _search_sokoban_solution(self, player_pos, boxes, targets, observation, env, valid_actions):
        """搜索推箱子解决方案"""
        # 简化的推箱子搜索逻辑
        # 找到最近的箱子
        if not boxes:
            return random.choice(valid_actions)
        
        closest_box = min(boxes, key=lambda box: abs(box[0] - player_pos[0]) + abs(box[1] - player_pos[1]))
        
        # 简单启发式：向最近的箱子移动
        dx = closest_box[0] - player_pos[0]
        dy = closest_box[1] - player_pos[1]
        
        # 转换为动作
        if abs(dx) > abs(dy):
            if dx > 0:
                preferred_action = 1  # DOWN
            else:
                preferred_action = 0  # UP
        else:
            if dy > 0:
                preferred_action = 3  # RIGHT
            else:
                preferred_action = 2  # LEFT
        
        return preferred_action if preferred_action in valid_actions else random.choice(valid_actions)
    
    def _search_gomoku_move(self, observation, env, valid_actions):
        """搜索五子棋最佳落子"""
        # 简化的五子棋搜索逻辑
        if hasattr(env, 'game') and hasattr(env.game, 'board'):
            board = env.game.board
            
            # 寻找可以连成多子的位置
            best_score = -1
            best_action = None
            
            for action in valid_actions:
                # 五子棋的动作是(row, col)元组
                if isinstance(action, tuple) and len(action) == 2:
                    row, col = action
                    score = self._evaluate_gomoku_position(board, row, col, self.player_id)
                    if score > best_score:
                        best_score = score
                        best_action = action
            
            return best_action
        
        return random.choice(valid_actions)
    
    def _evaluate_gomoku_position(self, board, row, col, player_id):
        """评估五子棋位置得分"""
        # 简单的位置评估函数
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        total_score = 0
        
        for dx, dy in directions:
            score = 1  # 自己这一颗子
            
            # 向前数
            for i in range(1, 5):
                new_row, new_col = row + i * dx, col + i * dy
                if (0 <= new_row < len(board) and 0 <= new_col < len(board[0]) and
                    board[new_row][new_col] == player_id):
                    score += 1
                else:
                    break
            
            # 向后数
            for i in range(1, 5):
                new_row, new_col = row - i * dx, col - i * dy
                if (0 <= new_row < len(board) and 0 <= new_col < len(board[0]) and
                    board[new_row][new_col] == player_id):
                    score += 1
                else:
                    break
            
            total_score += score ** 2  # 连子越多得分越高
        
        return total_score
    
    def get_info(self):
        """获取AI信息"""
        info = super().get_info()
        info.update({
            'search_algorithm': self.search_algorithm,
            'max_depth': self.max_depth,
            'description': f'搜索算法AI ({self.search_algorithm.upper()})'
        })
        return info
    
    def _enhanced_sokoban_heuristic(self, player_pos, boxes, targets, observation):
        """增强的推箱子启发式函数"""
        total_score = 0
        
        # 基础距离得分
        for box in boxes:
            if box not in targets:
                min_dist_to_target = min(abs(box[0] - t[0]) + abs(box[1] - t[1]) for t in targets)
                total_score += min_dist_to_target
        
        # 玩家到最近未完成箱子的距离
        incomplete_boxes = [box for box in boxes if box not in targets]
        if incomplete_boxes:
            min_player_to_box = min(abs(player_pos[0] - box[0]) + abs(player_pos[1] - box[1]) for box in incomplete_boxes)
            total_score += min_player_to_box * 0.1
        
        # 死锁惩罚
        deadlock_penalty = self._calculate_deadlock_penalty(boxes, targets, observation)
        total_score += deadlock_penalty
        
        # 聚集惩罚（箱子太聚集会难以操作）
        clustering_penalty = self._calculate_clustering_penalty(boxes, targets)
        total_score += clustering_penalty
        
        return total_score
    
    def _calculate_deadlock_penalty(self, boxes, targets, observation):
        """计算死锁惩罚"""
        penalty = 0
        board = observation['board'] if isinstance(observation, dict) else observation
        
        for box in boxes:
            if box not in targets:
                # 检查角落死锁
                if self._is_corner_position(box, board):
                    penalty += 1000
                
                # 检查墙边死锁
                if self._is_wall_edge_deadlock(box, board, targets):
                    penalty += 500
        
        return penalty
    
    def _is_corner_position(self, pos, board):
        """检查是否在角落位置"""
        row, col = pos
        corners = 0
        
        # 检查四个方向的墙
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if (new_row < 0 or new_row >= board.shape[0] or 
                new_col < 0 or new_col >= board.shape[1] or 
                board[new_row, new_col] == 1):
                corners += 1
        
        return corners >= 2
    
    def _is_wall_edge_deadlock(self, pos, board, targets):
        """检查墙边死锁"""
        row, col = pos
        
        # 如果贴着墙，检查沿墙方向是否有目标
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in directions:
            wall_row, wall_col = row + dr, col + dc
            
            # 如果这个方向是墙
            if (wall_row < 0 or wall_row >= board.shape[0] or 
                wall_col < 0 or wall_col >= board.shape[1] or 
                board[wall_row, wall_col] == 1):
                
                # 检查沿墙方向是否有目标
                perpendicular_dirs = [(dc, dr), (-dc, -dr)] if dr == 0 else [(dr, dc), (-dr, -dc)]
                has_target_along_wall = False
                
                for perp_dr, perp_dc in perpendicular_dirs:
                    for distance in range(1, min(board.shape)):
                        check_row = row + distance * perp_dr
                        check_col = col + distance * perp_dc
                        check_pos = (check_row, check_col)
                        
                        if check_pos in targets:
                            has_target_along_wall = True
                            break
                        
                        # 如果遇到墙就停止
                        if (check_row < 0 or check_row >= board.shape[0] or 
                            check_col < 0 or check_col >= board.shape[1] or 
                            board[check_row, check_col] == 1):
                            break
                
                if not has_target_along_wall:
                    return True
        
        return False
    
    def _calculate_clustering_penalty(self, boxes, targets):
        """计算聚集惩罚"""
        penalty = 0
        incomplete_boxes = [box for box in boxes if box not in targets]
        
        for i, box1 in enumerate(incomplete_boxes):
            for box2 in incomplete_boxes[i+1:]:
                distance = abs(box1[0] - box2[0]) + abs(box1[1] - box2[1])
                if distance == 1:  # 相邻的箱子
                    penalty += 50
                elif distance == 2:  # 很近的箱子
                    penalty += 20
        
        return penalty
