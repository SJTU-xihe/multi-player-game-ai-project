"""
MCTS Bot
使用蒙特卡洛树搜索算法
"""

import time
import random
import math
from typing import Dict, List, Tuple, Any, Optional
from agents.base_agent import BaseAgent
import config
import copy


class MCTSNode:
    """MCTS节点"""
    
    def __init__(self, game_state, parent=None, action=None, player_id=1):
        self.game_state = game_state
        self.parent = parent
        self.action = action  # 到达此节点的动作
        self.player_id = player_id
        self.children = {}  # action -> child_node
        self.visits = 0
        self.total_value = 0.0
        self.untried_actions = None
        self._initialize_untried_actions()
    
    def _initialize_untried_actions(self):
        """初始化未尝试的动作"""
        try:
            if hasattr(self.game_state, 'get_valid_actions'):
                self.untried_actions = list(self.game_state.get_valid_actions())
            else:
                self.untried_actions = []
        except:
            self.untried_actions = []
    
    def is_fully_expanded(self):
        """检查是否完全展开"""
        return len(self.untried_actions) == 0
    
    def is_terminal(self):
        """检查是否为终止节点"""
        try:
            return self.game_state.is_terminal()
        except:
            return True
    
    def get_winner(self):
        """获取获胜者"""
        try:
            return self.game_state.get_winner()
        except:
            return None
    
    def ucb1_value(self, exploration_weight=math.sqrt(2)):
        """计算UCB1值"""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.total_value / self.visits
        exploration = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def best_child(self, exploration_weight=math.sqrt(2)):
        """选择最佳子节点（UCB1策略）"""
        if not self.children:
            return None
        
        return max(self.children.values(), 
                  key=lambda child: child.ucb1_value(exploration_weight))
    
    def expand(self):
        """扩展节点"""
        if not self.untried_actions or self.is_terminal():
            return None
        
        # 选择一个未尝试的动作
        action = self.untried_actions.pop()
        
        try:
            # 克隆游戏状态并执行动作
            new_game_state = self.game_state.clone()
            new_game_state.step(action)
            
            # 创建新的子节点
            child_node = MCTSNode(new_game_state, parent=self, action=action, player_id=self.player_id)
            self.children[action] = child_node
            
            return child_node
        except Exception as e:
            # 如果动作失败，返回None
            return None
    
    def update(self, value):
        """更新节点统计信息"""
        self.visits += 1
        self.total_value += value
    
    def get_average_value(self):
        """获取平均值"""
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits


class MCTSBot(BaseAgent):
    """MCTS Bot"""
    
    def __init__(self, name: str = "MCTSBot", player_id: int = 1, 
                 simulation_count: int = 1000, timeout: float = 5.0):
        super().__init__(name, player_id)
        self.simulation_count = simulation_count
        self.timeout = timeout
        self.exploration_weight = math.sqrt(2)
        
        # 从配置获取参数
        try:
            ai_config = config.AI_CONFIGS.get('mcts', {})
            self.simulation_count = ai_config.get('simulation_count', simulation_count)
            self.timeout = ai_config.get('timeout', timeout)
        except:
            pass
    
    def get_action(self, observation: Any, env: Any) -> Any:
        """
        使用MCTS选择动作
        
        Args:
            observation: 当前观察
            env: 环境对象
            
        Returns:
            选择的动作
        """
        start_time = time.time()
        
        # 获取有效动作
        valid_actions = env.get_valid_actions()
        
        if not valid_actions:
            return None
        
        if len(valid_actions) == 1:
            return valid_actions[0]
        
        # 创建根节点
        root = MCTSNode(env.game.clone(), player_id=self.player_id)
        
        simulations = 0
        
        # MCTS主循环
        while simulations < self.simulation_count and time.time() - start_time < self.timeout:
            # 1. 选择 (Selection)
            node = self._select(root)
            
            # 2. 扩展 (Expansion)
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()
                if node is None:
                    continue
            
            # 3. 模拟 (Simulation)
            value = self._simulate(node)
            
            # 4. 反向传播 (Backpropagation)
            self._backpropagate(node, value)
            
            simulations += 1
        
        # 选择访问次数最多的子节点
        if not root.children:
            return random.choice(valid_actions)
        
        best_action = max(root.children.keys(), 
                         key=lambda action: root.children[action].visits)
        
        print(f"MCTSBot: {simulations} simulations in {time.time() - start_time:.3f}s")
        
        # 更新统计
        move_time = time.time() - start_time
        self.total_moves += 1
        self.total_time += move_time
        
        return best_action
    
    def _select(self, node):
        """选择阶段：使用UCB1策略选择到叶子节点"""
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child(self.exploration_weight)
            if node is None:
                break
        return node
    
    def _simulate(self, node):
        """模拟阶段：从当前节点随机模拟到游戏结束"""
        if node is None:
            return 0
        
        try:
            # 克隆游戏状态
            game_state = node.game_state.clone()
            
            # 随机模拟
            simulation_depth = 0
            max_simulation_depth = 50  # 防止无限循环
            
            while not game_state.is_terminal() and simulation_depth < max_simulation_depth:
                valid_actions = game_state.get_valid_actions()
                if not valid_actions:
                    break
                
                # 使用改进的模拟策略
                action = self._select_simulation_action(game_state, valid_actions)
                game_state.step(action)
                simulation_depth += 1
            
            # 评估最终状态
            return self._evaluate_final_state(game_state)
            
        except Exception as e:
            return 0
    
    def _select_simulation_action(self, game_state, valid_actions):
        """改进的模拟策略：不完全随机，有一定启发式"""
        if len(valid_actions) == 1:
            return valid_actions[0]
        
        try:
            # 80%的时间使用启发式策略，20%的时间随机选择
            if random.random() < 0.8:
                return self._heuristic_action_selection(game_state, valid_actions)
            else:
                return random.choice(valid_actions)
        except:
            return random.choice(valid_actions)
    
    def _heuristic_action_selection(self, game_state, valid_actions):
        """启发式动作选择"""
        try:
            state = game_state.get_state()
            current_player = state.get('current_player', self.player_id)
            
            # 获取蛇的信息
            if current_player == 1:
                my_snake = state.get('snake1', [])
            else:
                my_snake = state.get('snake2', [])
            
            if not my_snake:
                return random.choice(valid_actions)
            
            my_head = my_snake[0]
            foods = state.get('foods', [])
            
            best_action = None
            best_score = float('-inf')
            
            for action in valid_actions:
                score = 0
                
                # 计算移动后的新头部位置
                new_head = (my_head[0] + action[0], my_head[1] + action[1])
                
                # 1. 避免边界
                board_size = getattr(game_state, 'board_size', 20)
                if (new_head[0] < 0 or new_head[0] >= board_size or 
                    new_head[1] < 0 or new_head[1] >= board_size):
                    score -= 1000
                    continue
                
                # 2. 避免撞到自己或对手
                all_snake_positions = []
                if 'snake1' in state:
                    all_snake_positions.extend(state['snake1'])
                if 'snake2' in state:
                    all_snake_positions.extend(state['snake2'])
                
                if new_head in all_snake_positions:
                    score -= 1000
                    continue
                
                # 3. 倾向于靠近食物
                if foods:
                    min_food_dist = min(abs(new_head[0] - food[0]) + abs(new_head[1] - food[1]) 
                                      for food in foods)
                    score -= min_food_dist * 10
                
                # 4. 避免靠近边界
                border_dist = min(new_head[0], new_head[1], 
                                board_size - 1 - new_head[0], board_size - 1 - new_head[1])
                score += border_dist * 5
                
                # 5. 计算可达空间（简化版）
                reachable = self._count_reachable_spaces(game_state, new_head, all_snake_positions)
                score += reachable * 2
                
                if score > best_score:
                    best_score = score
                    best_action = action
            
            return best_action if best_action is not None else random.choice(valid_actions)
            
        except:
            return random.choice(valid_actions)
    
    def _count_reachable_spaces(self, game_state, start_pos, obstacles, max_depth=5):
        """计算可达空间数量（简化版BFS）"""
        try:
            board_size = getattr(game_state, 'board_size', 20)
            visited = set()
            queue = [start_pos]
            visited.add(start_pos)
            count = 0
            
            while queue and count < max_depth:
                pos = queue.pop(0)
                count += 1
                
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_pos = (pos[0] + dx, pos[1] + dy)
                    
                    if (0 <= new_pos[0] < board_size and 
                        0 <= new_pos[1] < board_size and 
                        new_pos not in visited and 
                        new_pos not in obstacles):
                        visited.add(new_pos)
                        queue.append(new_pos)
                        
            return len(visited)
        except:
            return 1
    
    def _evaluate_final_state(self, game_state):
        """评估最终状态"""
        try:
            winner = game_state.get_winner()
            
            if winner == self.player_id:
                return 1.0  # 我赢了
            elif winner is not None:
                return -1.0  # 我输了
            else:
                # 平局或游戏未结束，使用启发式评估
                return self._heuristic_evaluation(game_state)
        except:
            return 0.0
    
    def _heuristic_evaluation(self, game_state):
        """启发式评估函数"""
        try:
            state = game_state.get_state()
            
            # 基本存活检查
            alive1 = state.get('alive1', True)
            alive2 = state.get('alive2', True)
            
            if self.player_id == 1:
                if not alive1:
                    return -1.0
                elif not alive2:
                    return 1.0
            else:
                if not alive2:
                    return -1.0
                elif not alive1:
                    return 1.0
            
            # 长度比较
            snake1 = state.get('snake1', [])
            snake2 = state.get('snake2', [])
            
            my_length = len(snake1) if self.player_id == 1 else len(snake2)
            enemy_length = len(snake2) if self.player_id == 1 else len(snake1)
            
            length_diff = my_length - enemy_length
            return length_diff * 0.1  # 归一化到[-1, 1]范围
            
        except:
            return 0.0
    
    def _backpropagate(self, node, value):
        """反向传播阶段：更新从叶子节点到根节点路径上的所有节点"""
        while node is not None:
            # 对于对手的回合，需要反转价值
            current_player = getattr(node.game_state, 'current_player', self.player_id)
            if current_player != self.player_id:
                node.update(-value)
            else:
                node.update(value)
            node = node.parent
    
    def reset(self):
        """重置MCTS Bot"""
        super().reset()
    
    def get_info(self) -> Dict[str, Any]:
        """获取MCTS Bot信息"""
        info = super().get_info()
        info.update({
            'type': 'MCTS',
            'description': '使用蒙特卡洛树搜索的Bot',
            'strategy': f'MCTS with {self.simulation_count} simulations',
            'timeout': self.timeout,
            'exploration_weight': self.exploration_weight
        })
        return info 