from agents.base_agent import BaseAgent
import copy
import time
import math

class MinimaxBot(BaseAgent):
    def __init__(self, name="MinimaxBot", player_id=1, max_depth=3, timeout=5.0):
        super().__init__(name, player_id)
        self.max_depth = max_depth
        self.timeout = timeout  # 每步最大思考时间（秒）
        self.nodes_searched = 0
        self.start_time = 0
        self.transposition_table = {}  # 状态缓存表

    def get_action(self, observation, env):
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            return None
        
        if len(valid_actions) == 1:
            return valid_actions[0]
            
        self.start_time = time.time()
        self.nodes_searched = 0
        self.transposition_table.clear()  # 清空缓存
        
        best_score = float('-inf')
        best_action = valid_actions[0]
        
        # 迭代加深搜索，从深度1开始逐步增加
        for depth in range(1, self.max_depth + 1):
            if self._is_timeout():
                break
                
            current_best_score = float('-inf')
            current_best_action = valid_actions[0]
            
            # 对每个动作进行minimax搜索
            for action in valid_actions:
                if self._is_timeout():
                    break
                    
                try:
                    # 克隆游戏状态并执行动作
                    game_copy = env.game.clone()
                    game_copy.step(action)
                    
                    # 使用alpha-beta剪枝进行搜索
                    score = self.minimax_ab(game_copy, depth - 1, False, 
                                          float('-inf'), float('inf'))
                    
                    if score > current_best_score:
                        current_best_score = score
                        current_best_action = action
                        
                except Exception as e:
                    # 如果动作执行失败，给最低分
                    continue
            
            # 如果这一层搜索完成，更新最佳选择
            if not self._is_timeout():
                best_score = current_best_score
                best_action = current_best_action
                
        print(f"MinimaxBot searched {self.nodes_searched} nodes in {time.time() - self.start_time:.3f}s")
        return best_action

    def minimax_ab(self, game, depth, maximizing_player, alpha, beta):
        """Alpha-Beta剪枝的Minimax算法"""
        self.nodes_searched += 1
        
        # 超时检查
        if self._is_timeout():
            return self.evaluate_position(game)
            
        # 游戏状态哈希（简单实现）
        state_key = self._get_state_hash(game)
        if state_key in self.transposition_table and depth <= self.transposition_table[state_key]['depth']:
            return self.transposition_table[state_key]['score']
            
        # 终止条件
        if depth == 0 or game.is_terminal():
            score = self.evaluate_position(game)
            self.transposition_table[state_key] = {'score': score, 'depth': depth}
            return score
        
        valid_actions = game.get_valid_actions()
        if not valid_actions:
            score = self.evaluate_position(game)
            self.transposition_table[state_key] = {'score': score, 'depth': depth}
            return score
            
        if maximizing_player:
            max_eval = float('-inf')
            for action in valid_actions:
                if self._is_timeout():
                    break
                    
                try:
                    game_copy = game.clone()
                    game_copy.step(action)
                    eval_score = self.minimax_ab(game_copy, depth - 1, False, alpha, beta)
                    max_eval = max(max_eval, eval_score)
                    alpha = max(alpha, eval_score)
                    
                    # Alpha-beta剪枝
                    if beta <= alpha:
                        break
                except:
                    continue
                    
            self.transposition_table[state_key] = {'score': max_eval, 'depth': depth}
            return max_eval
        else:
            min_eval = float('inf')
            for action in valid_actions:
                if self._is_timeout():
                    break
                    
                try:
                    game_copy = game.clone()
                    game_copy.step(action)
                    eval_score = self.minimax_ab(game_copy, depth - 1, True, alpha, beta)
                    min_eval = min(min_eval, eval_score)
                    beta = min(beta, eval_score)
                    
                    # Alpha-beta剪枝
                    if beta <= alpha:
                        break
                except:
                    continue
                    
            self.transposition_table[state_key] = {'score': min_eval, 'depth': depth}
            return min_eval

    def evaluate_position(self, game):
        """改进的位置评估函数"""
        # 基本胜负判断
        winner = game.get_winner()
        if winner == self.player_id:
            return 10000  # 我赢了
        elif winner is not None:
            return -10000  # 我输了
        elif game.is_terminal():
            return 0  # 平局
            
        try:
            state = game.get_state()
            my_snake = state['snake1'] if self.player_id == 1 else state['snake2']
            enemy_snake = state['snake2'] if self.player_id == 1 else state['snake1']
            foods = state['foods']
            alive1 = state.get('alive1', True)
            alive2 = state.get('alive2', True)
            
            # 检查存活状态
            if self.player_id == 1 and not alive1:
                return -10000
            elif self.player_id == 2 and not alive2:
                return -10000
            elif self.player_id == 1 and not alive2:
                return 10000
            elif self.player_id == 2 and not alive1:
                return 10000
            
            score = 0
            
            # 1. 蛇长度优势（最重要）
            my_length = len(my_snake) if my_snake else 0
            enemy_length = len(enemy_snake) if enemy_snake else 0
            score += (my_length - enemy_length) * 100
            
            if my_snake and len(my_snake) > 0:
                my_head = my_snake[0]
                board_size = getattr(game, 'board_size', 20)
                
                # 2. 到食物的距离（鼓励吃食物）
                if foods:
                    min_food_dist = min(self._manhattan_distance(my_head, food) for food in foods)
                    score -= min_food_dist * 10  # 距离食物越近越好
                
                # 3. 避免边界（保持在中央区域）
                border_penalty = 0
                if my_head[0] <= 1 or my_head[0] >= board_size - 2:
                    border_penalty += 20
                if my_head[1] <= 1 or my_head[1] >= board_size - 2:
                    border_penalty += 20
                score -= border_penalty
                
                # 4. 空间控制（计算可达空间）
                my_space = self._calculate_reachable_space(game, my_head, my_snake + enemy_snake)
                enemy_space = 0
                if enemy_snake and len(enemy_snake) > 0:
                    enemy_head = enemy_snake[0]
                    enemy_space = self._calculate_reachable_space(game, enemy_head, my_snake + enemy_snake)
                
                score += (my_space - enemy_space) * 5
                
                # 5. 避免死路（简单检查）
                danger_score = self._calculate_danger(game, my_head, my_snake + enemy_snake)
                score -= danger_score * 50
                
                # 6. 与对手的距离（保持适当距离）
                if enemy_snake and len(enemy_snake) > 0:
                    enemy_head = enemy_snake[0]
                    distance_to_enemy = self._manhattan_distance(my_head, enemy_head)
                    if distance_to_enemy < 3:  # 太近很危险
                        score -= (3 - distance_to_enemy) * 30
                    elif distance_to_enemy > 10:  # 太远可能错失机会
                        score -= (distance_to_enemy - 10) * 5
            
            return score
            
        except Exception as e:
            return 0

    def _manhattan_distance(self, pos1, pos2):
        """计算曼哈顿距离"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _calculate_reachable_space(self, game, start_pos, obstacles, max_depth=10):
        """计算从起始位置能到达的空间大小（BFS）"""
        if max_depth <= 0:
            return 1
            
        board_size = getattr(game, 'board_size', 20)
        visited = set()
        queue = [start_pos]
        visited.add(start_pos)
        count = 0
        
        while queue and count < max_depth:
            pos = queue.pop(0)
            count += 1
            
            # 检查四个方向
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_pos = (pos[0] + dx, pos[1] + dy)
                
                # 检查边界
                if (new_pos[0] < 0 or new_pos[0] >= board_size or 
                    new_pos[1] < 0 or new_pos[1] >= board_size):
                    continue
                    
                # 检查是否已访问或有障碍物
                if new_pos not in visited and new_pos not in obstacles:
                    visited.add(new_pos)
                    queue.append(new_pos)
                    
        return len(visited)

    def _calculate_danger(self, game, head_pos, obstacles):
        """计算位置的危险程度"""
        board_size = getattr(game, 'board_size', 20)
        danger = 0
        
        # 检查四个方向的危险
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        blocked_directions = 0
        
        for dx, dy in directions:
            new_pos = (head_pos[0] + dx, head_pos[1] + dy)
            
            # 检查是否撞墙或撞蛇
            if (new_pos[0] < 0 or new_pos[0] >= board_size or 
                new_pos[1] < 0 or new_pos[1] >= board_size or 
                new_pos in obstacles):
                blocked_directions += 1
        
        # 被堵住的方向越多越危险
        if blocked_directions >= 3:
            danger += 100  # 非常危险
        elif blocked_directions == 2:
            danger += 50   # 比较危险
        elif blocked_directions == 1:
            danger += 10   # 稍微危险
            
        return danger

    def _get_state_hash(self, game):
        """生成游戏状态的简单哈希值"""
        try:
            state = game.get_state()
            # 简单的状态哈希：基于蛇的位置和当前玩家
            snake1_str = str(sorted(state['snake1'])) if state['snake1'] else "[]"
            snake2_str = str(sorted(state['snake2'])) if state['snake2'] else "[]"
            foods_str = str(sorted(state['foods'])) if state['foods'] else "[]"
            return f"{snake1_str}_{snake2_str}_{foods_str}_{state['current_player']}"
        except:
            return str(time.time())  # 如果哈希失败，使用时间戳

    def _is_timeout(self):
        """检查是否超时"""
        return time.time() - self.start_time > self.timeout

