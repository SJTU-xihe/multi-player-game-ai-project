"""
贪吃蛇专用AI智能体
实现A*寻路算法、安全性评估和对手行为预测
"""

import random
import heapq
import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from agents.base_agent import BaseAgent


class SnakeAI(BaseAgent):
    """高级贪吃蛇AI智能体"""
    
    def __init__(self, name="SnakeAI", player_id=1):
        super().__init__(name, player_id)
        self.path_cache = {}  # 路径缓存
        self.opponent_history = []  # 对手历史动作
        self.max_history = 10  # 保存最近10步
    
    def get_action(self, observation, env):
        """获取动作"""
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            return None
        
        if len(valid_actions) == 1:
            return valid_actions[0]
        
        game = env.game
        
        # 获取当前蛇的信息
        if self.player_id == 1:
            my_snake = game.snake1
            enemy_snake = game.snake2
            my_direction = game.direction1
        else:
            my_snake = game.snake2
            enemy_snake = game.snake1
            my_direction = game.direction2
        
        if not my_snake:
            return random.choice(valid_actions)
        
        my_head = my_snake[0]
        
        # 更新对手历史
        self._update_opponent_history(enemy_snake)
        
        # 预测对手下一步行为
        predicted_enemy_positions = self._predict_opponent_moves(enemy_snake, game)
        
        # 评估每个动作的安全性和价值
        action_scores = {}
        
        for action in valid_actions:
            score = self._evaluate_action(action, my_head, my_snake, enemy_snake, 
                                        game, predicted_enemy_positions)
            action_scores[action] = score
        
        # 选择最佳动作
        best_action = max(action_scores.keys(), key=lambda a: action_scores[a])
        
        return best_action
    
    def _update_opponent_history(self, enemy_snake):
        """更新对手历史动作"""
        if enemy_snake and len(enemy_snake) > 0:
            current_head = enemy_snake[0]
            
            # 如果有历史记录，计算移动方向
            if len(self.opponent_history) > 0:
                last_head = self.opponent_history[-1]
                if last_head != current_head:
                    direction = (current_head[0] - last_head[0], current_head[1] - last_head[1])
                    # 保存历史，但限制长度
                    if len(self.opponent_history) >= self.max_history:
                        self.opponent_history.pop(0)
            
            self.opponent_history.append(current_head)
    
    def _predict_opponent_moves(self, enemy_snake, game) -> List[Tuple[int, int]]:
        """预测对手可能的移动位置"""
        if not enemy_snake or len(enemy_snake) == 0:
            return []
        
        enemy_head = enemy_snake[0]
        predicted_positions = []
        
        # 获取对手的有效动作
        valid_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for direction in valid_directions:
            new_pos = (enemy_head[0] + direction[0], enemy_head[1] + direction[1])
            
            # 检查是否在边界内
            if (0 <= new_pos[0] < game.board_size and 
                0 <= new_pos[1] < game.board_size):
                
                # 检查是否撞到蛇身
                if (new_pos not in enemy_snake[:-1] and 
                    new_pos not in game.snake1[:-1] and 
                    new_pos not in game.snake2[:-1]):
                    predicted_positions.append(new_pos)
        
        return predicted_positions
    
    def _evaluate_action(self, action, my_head, my_snake, enemy_snake, game, 
                        predicted_enemy_positions) -> float:
        """评估动作的综合得分"""
        new_head = (my_head[0] + action[0], my_head[1] + action[1])
        
        # 基础安全检查
        safety_score = self._calculate_safety_score(new_head, my_snake, enemy_snake, 
                                                   game, predicted_enemy_positions)
        
        if safety_score < 0:  # 不安全的动作
            return safety_score
        
        # 食物相关得分
        food_score = self._calculate_food_score(new_head, game.foods, game)
        
        # 空间控制得分
        space_score = self._calculate_space_score(new_head, my_snake, enemy_snake, game)
        
        # 对手距离得分
        enemy_distance_score = self._calculate_enemy_distance_score(new_head, enemy_snake)
        
        # 中心位置得分
        center_score = self._calculate_center_score(new_head, game.board_size)
        
        # 综合得分
        total_score = (safety_score * 10.0 +    # 安全性最重要
                      food_score * 5.0 +        # 食物重要性
                      space_score * 3.0 +       # 空间控制
                      enemy_distance_score * 2.0 + # 与对手距离
                      center_score * 1.0)       # 中心位置
        
        return total_score
    
    def _calculate_safety_score(self, new_head, my_snake, enemy_snake, game, 
                               predicted_enemy_positions) -> float:
        """计算安全性得分"""
        # 检查边界
        if (new_head[0] < 0 or new_head[0] >= game.board_size or
            new_head[1] < 0 or new_head[1] >= game.board_size):
            return -1000.0
        
        # 检查撞到自己
        if new_head in my_snake[:-1]:
            return -1000.0
        
        # 检查撞到对手
        if enemy_snake and new_head in enemy_snake[:-1]:
            return -1000.0
        
        # 检查是否会与对手头部碰撞
        if new_head in predicted_enemy_positions:
            return -500.0  # 高风险，但不是立即死亡
        
        # 计算死路风险
        escape_routes = self._count_escape_routes(new_head, my_snake, enemy_snake, game)
        
        if escape_routes == 0:
            return -800.0  # 死路
        elif escape_routes == 1:
            return -200.0  # 危险
        elif escape_routes == 2:
            return 50.0    # 一般
        else:
            return 100.0   # 安全
    
    def _count_escape_routes(self, pos, my_snake, enemy_snake, game, depth=3) -> int:
        """计算从给定位置的逃生路线数量"""
        if depth <= 0:
            return 1
        
        count = 0
        for direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_pos = (pos[0] + direction[0], pos[1] + direction[1])
            
            # 检查边界
            if (next_pos[0] < 0 or next_pos[0] >= game.board_size or
                next_pos[1] < 0 or next_pos[1] >= game.board_size):
                continue
            
            # 检查障碍物
            all_obstacles = set()
            if my_snake:
                all_obstacles.update(my_snake[:-1])
            if enemy_snake:
                all_obstacles.update(enemy_snake[:-1])
            
            if next_pos not in all_obstacles:
                count += self._count_escape_routes(next_pos, my_snake, enemy_snake, game, depth - 1)
        
        return count
    
    def _calculate_food_score(self, new_head, foods, game) -> float:
        """计算食物相关得分"""
        if not foods:
            return 0.0
        
        # 使用A*寻找到最近食物的路径
        nearest_food = self._find_nearest_food(new_head, foods)
        path = self._a_star_pathfinding(new_head, nearest_food, game)
        
        if path:
            # 路径越短得分越高
            distance = len(path) - 1
            return max(0, 100 - distance * 5)
        else:
            # 如果找不到路径，使用曼哈顿距离
            min_distance = min(abs(new_head[0] - food[0]) + abs(new_head[1] - food[1]) 
                             for food in foods)
            return max(0, 50 - min_distance * 3)
    
    def _find_nearest_food(self, head, foods) -> Tuple[int, int]:
        """找到最近的食物"""
        min_distance = float('inf')
        nearest_food = foods[0]
        
        for food in foods:
            distance = abs(head[0] - food[0]) + abs(head[1] - food[1])
            if distance < min_distance:
                min_distance = distance
                nearest_food = food
        
        return nearest_food
    
    def _a_star_pathfinding(self, start, goal, game) -> Optional[List[Tuple[int, int]]]:
        """A*寻路算法"""
        # 缓存键
        cache_key = (start, goal, hash(str(game.snake1 + game.snake2)))
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        def get_neighbors(pos):
            x, y = pos
            neighbors = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < game.board_size and 0 <= ny < game.board_size):
                    # 检查障碍物（允许通过尾部，因为它会移动）
                    obstacles = set()
                    if game.snake1:
                        obstacles.update(game.snake1[:-1])
                    if game.snake2:
                        obstacles.update(game.snake2[:-1])
                    
                    if (nx, ny) not in obstacles:
                        neighbors.append((nx, ny))
            return neighbors
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        visited = set()
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current in visited:
                continue
            visited.add(current)
            
            if current == goal:
                # 重构路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                
                # 缓存结果
                self.path_cache[cache_key] = path
                return path
            
            for neighbor in get_neighbors(current):
                if neighbor in visited:
                    continue
                
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # 清理缓存以避免内存泄漏
        if len(self.path_cache) > 100:
            self.path_cache.clear()
        
        return None
    
    def _calculate_space_score(self, new_head, my_snake, enemy_snake, game) -> float:
        """计算空间控制得分"""
        # 使用BFS计算可达空间
        reachable_spaces = self._bfs_reachable_spaces(new_head, my_snake, enemy_snake, game)
        return min(reachable_spaces * 2, 100)  # 限制最大得分
    
    def _bfs_reachable_spaces(self, start, my_snake, enemy_snake, game, max_depth=15) -> int:
        """使用BFS计算可达空间数量"""
        visited = set()
        queue = [(start, 0)]
        visited.add(start)
        
        while queue:
            pos, depth = queue.pop(0)
            
            if depth >= max_depth:
                continue
            
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                next_pos = (pos[0] + dx, pos[1] + dy)
                
                if (0 <= next_pos[0] < game.board_size and 
                    0 <= next_pos[1] < game.board_size and 
                    next_pos not in visited):
                    
                    # 检查障碍物
                    obstacles = set()
                    if my_snake:
                        obstacles.update(my_snake[:-1])
                    if enemy_snake:
                        obstacles.update(enemy_snake[:-1])
                    
                    if next_pos not in obstacles:
                        visited.add(next_pos)
                        queue.append((next_pos, depth + 1))
        
        return len(visited)
    
    def _calculate_enemy_distance_score(self, new_head, enemy_snake) -> float:
        """计算与对手距离得分"""
        if not enemy_snake or len(enemy_snake) == 0:
            return 0.0
        
        enemy_head = enemy_snake[0]
        distance = abs(new_head[0] - enemy_head[0]) + abs(new_head[1] - enemy_head[1])
        
        # 保持适当距离：太近危险，太远可能错失机会
        if distance < 3:
            return -distance * 10  # 太近，惩罚
        elif distance <= 6:
            return distance * 5    # 适中距离，奖励
        else:
            return max(0, 30 - distance)  # 距离过远，轻微惩罚
    
    def _calculate_center_score(self, new_head, board_size) -> float:
        """计算中心位置得分"""
        center = board_size // 2
        distance_to_center = abs(new_head[0] - center) + abs(new_head[1] - center)
        return max(0, 20 - distance_to_center * 2)


class SmartSnakeAI(SnakeAI):
    """更智能的贪吃蛇AI，继承自SnakeAI并添加更多功能"""
    
    def __init__(self, name="SmartSnakeAI", player_id=1):
        super().__init__(name, player_id)
        self.aggressive_mode = False  # 侵略模式
        self.defensive_mode = False   # 防守模式
    
    def get_action(self, observation, env):
        """智能决策，根据情况调整策略"""
        # 分析局势
        self._analyze_game_situation(env.game)
        
        # 根据模式调整权重
        action = super().get_action(observation, env)
        
        return action
    
    def _analyze_game_situation(self, game):
        """分析游戏局势，决定策略"""
        if self.player_id == 1:
            my_length = len(game.snake1) if game.snake1 else 0
            enemy_length = len(game.snake2) if game.snake2 else 0
        else:
            my_length = len(game.snake2) if game.snake2 else 0
            enemy_length = len(game.snake1) if game.snake1 else 0
        
        # 根据长度差决定策略
        length_diff = my_length - enemy_length
        
        if length_diff > 3:
            self.aggressive_mode = True
            self.defensive_mode = False
        elif length_diff < -3:
            self.aggressive_mode = False
            self.defensive_mode = True
        else:
            self.aggressive_mode = False
            self.defensive_mode = False
    
    def _evaluate_action(self, action, my_head, my_snake, enemy_snake, game, 
                        predicted_enemy_positions) -> float:
        """重写评估函数，根据模式调整权重"""
        base_score = super()._evaluate_action(action, my_head, my_snake, enemy_snake, 
                                            game, predicted_enemy_positions)
        
        if self.aggressive_mode:
            # 侵略模式：更关注食物和进攻
            new_head = (my_head[0] + action[0], my_head[1] + action[1])
            food_bonus = self._calculate_food_score(new_head, game.foods, game) * 0.5
            return base_score + food_bonus
        
        elif self.defensive_mode:
            # 防守模式：更关注安全性
            safety_bonus = self._calculate_safety_score(new_head, my_snake, enemy_snake, 
                                                       game, predicted_enemy_positions) * 0.3
            return base_score + safety_bonus
        
        return base_score 