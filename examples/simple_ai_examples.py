"""
简单AI实现示例
展示如何实现基于规则、贪心算法等基础AI
"""

import random
import sys
import os
from typing import List, Tuple, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseAgent


class ImprovedRandomBot(BaseAgent):
    """改进的随机AI - 避免明显的错误决策"""
    
    def __init__(self, name: str = "ImprovedRandomBot", player_id: int = 1):
        super().__init__(name, player_id)
    
    def get_action(self, observation: Any, env: Any) -> Any:
        """改进的随机选择 - 过滤掉明显不好的动作"""
        valid_actions = env.get_valid_actions()
        
        if not valid_actions:
            return None
        
        # 过滤掉明显不利的动作
        good_actions = self._filter_bad_actions(valid_actions, observation, env)
        
        if good_actions:
            return random.choice(good_actions)
        else:
            return random.choice(valid_actions)
    
    def _filter_bad_actions(self, actions: List[Any], observation: Any, env: Any) -> List[Any]:
        """过滤掉明显不好的动作"""
        good_actions = []
        
        for action in actions:
            # 简单检查：避免立即失败的动作
            if not self._is_immediately_bad(action, observation, env):
                good_actions.append(action)
        
        return good_actions
    
    def _is_immediately_bad(self, action: Any, observation: Any, env: Any) -> bool:
        """检查动作是否会立即导致失败"""
        # 这里可以根据具体游戏实现检查逻辑
        # 例如：检查是否会撞墙、撞自己等
        return False


class RuleBasedGomokuBot(BaseAgent):
    """基于规则的五子棋AI"""
    
    def __init__(self, name: str = "RuleBasedGomokuBot", player_id: int = 1):
        super().__init__(name, player_id)
    
    def get_action(self, observation: Any, env: Any) -> Tuple[int, int]:
        """基于规则的决策"""
        valid_actions = env.get_valid_actions()
        
        # 处理不同格式的观察值
        if isinstance(observation, dict):
            board = observation['board']
        else:
            # 假设observation直接是棋盘数组
            board = observation
        
        # 规则1: 如果能获胜，立即获胜
        winning_move = self._find_winning_move(valid_actions, board, self.player_id)
        if winning_move:
            return winning_move
        
        # 规则2: 如果对手能获胜，阻止对手
        opponent_id = 3 - self.player_id
        blocking_move = self._find_winning_move(valid_actions, board, opponent_id)
        if blocking_move:
            return blocking_move
        
        # 规则3: 寻找能形成威胁的位置
        threat_move = self._find_threat_move(valid_actions, board, self.player_id)
        if threat_move:
            return threat_move
        
        # 规则4: 选择中心附近的位置
        center_move = self._find_center_move(valid_actions, board)
        if center_move:
            return center_move
        
        # 规则5: 随机选择
        return random.choice(valid_actions)
    
    def _find_winning_move(self, actions: List[Tuple[int, int]], board: Any, player: int) -> Tuple[int, int]:
        """寻找获胜动作"""
        for action in actions:
            row, col = action
            if self._check_win_after_move(board, row, col, player):
                return action
        return None
    
    def _check_win_after_move(self, board: Any, row: int, col: int, player: int) -> bool:
        """检查放置棋子后是否获胜"""
        # 临时放置棋子
        original = board[row, col]
        board[row, col] = player
        
        # 检查四个方向
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        win = False
        
        for dr, dc in directions:
            count = 1  # 当前位置
            
            # 正方向
            r, c = row + dr, col + dc
            while (0 <= r < board.shape[0] and 0 <= c < board.shape[1] and 
                   board[r, c] == player):
                count += 1
                r += dr
                c += dc
            
            # 负方向
            r, c = row - dr, col - dc
            while (0 <= r < board.shape[0] and 0 <= c < board.shape[1] and 
                   board[r, c] == player):
                count += 1
                r -= dr
                c -= dc
            
            if count >= 5:
                win = True
                break
        
        # 恢复原状
        board[row, col] = original
        return win
    
    def _find_threat_move(self, actions: List[Tuple[int, int]], board: Any, player: int) -> Tuple[int, int]:
        """寻找能形成威胁的动作"""
        best_action = None
        max_threat = 0
        
        for action in actions:
            threat_level = self._calculate_threat_level(board, action, player)
            if threat_level > max_threat:
                max_threat = threat_level
                best_action = action
        
        return best_action if max_threat > 0 else None
    
    def _calculate_threat_level(self, board: Any, action: Tuple[int, int], player: int) -> int:
        """计算威胁等级"""
        row, col = action
        threat = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = self._count_consecutive_in_direction(board, row, col, dr, dc, player)
            if count >= 3:
                threat += count * 2
            elif count >= 2:
                threat += count
        
        return threat
    
    def _count_consecutive_in_direction(self, board: Any, row: int, col: int, 
                                      dr: int, dc: int, player: int) -> int:
        """计算某方向连续棋子数"""
        count = 1  # 当前位置
        
        # 正方向
        r, c = row + dr, col + dc
        while (0 <= r < board.shape[0] and 0 <= c < board.shape[1] and 
               board[r, c] == player):
            count += 1
            r += dr
            c += dc
        
        # 负方向
        r, c = row - dr, col - dc
        while (0 <= r < board.shape[0] and 0 <= c < board.shape[1] and 
               board[r, c] == player):
            count += 1
            r -= dr
            c -= dc
        
        return count
    
    def _find_center_move(self, actions: List[Tuple[int, int]], board: Any) -> Tuple[int, int]:
        """寻找靠近中心的动作"""
        center = board.shape[0] // 2
        min_distance = float('inf')
        best_action = None
        
        for action in actions:
            row, col = action
            distance = abs(row - center) + abs(col - center)
            if distance < min_distance:
                min_distance = distance
                best_action = action
        
        return best_action


class GreedySnakeBot(BaseAgent):
    """贪心算法贪吃蛇AI"""
    
    def __init__(self, name: str = "GreedySnakeBot", player_id: int = 1):
        super().__init__(name, player_id)
    
    def get_action(self, observation: Any, env: Any) -> Tuple[int, int]:
        """贪心策略：总是朝最近的食物移动"""
        valid_actions = env.get_valid_actions()
        
        if not valid_actions:
            return (0, 1)  # 默认向右
        
        # 获取当前状态
        snake = observation['snake1'] if self.player_id == 1 else observation['snake2']
        foods = observation['foods']
        head = snake[0]
        
        if not foods:
            return random.choice(valid_actions)
        
        # 找到最近的食物
        nearest_food = self._find_nearest_food(head, foods)
        
        # 计算朝向最近食物的最佳方向
        best_action = self._choose_direction_to_target(head, nearest_food, valid_actions)
        
        return best_action if best_action else random.choice(valid_actions)
    
    def _find_nearest_food(self, head: Tuple[int, int], foods: List[Tuple[int, int]]) -> Tuple[int, int]:
        """找到最近的食物"""
        min_distance = float('inf')
        nearest_food = foods[0]
        
        for food in foods:
            distance = abs(head[0] - food[0]) + abs(head[1] - food[1])
            if distance < min_distance:
                min_distance = distance
                nearest_food = food
        
        return nearest_food
    
    def _choose_direction_to_target(self, head: Tuple[int, int], target: Tuple[int, int], 
                                  valid_actions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """选择朝向目标的方向"""
        target_row, target_col = target
        head_row, head_col = head
        
        # 计算需要移动的方向
        dr = target_row - head_row
        dc = target_col - head_col
        
        # 优先级：距离目标更近的方向
        preferred_actions = []
        
        if dr > 0:
            preferred_actions.append((1, 0))  # 向下
        elif dr < 0:
            preferred_actions.append((-1, 0))  # 向上
        
        if dc > 0:
            preferred_actions.append((0, 1))  # 向右
        elif dc < 0:
            preferred_actions.append((0, -1))  # 向左
        
        # 选择第一个有效的优先方向
        for action in preferred_actions:
            if action in valid_actions:
                return action
        
        return None


class SearchBasedBot(BaseAgent):
    """基于搜索的AI（简化BFS示例）"""
    
    def __init__(self, name: str = "SearchBasedBot", player_id: int = 1, max_depth: int = 3):
        super().__init__(name, player_id)
        self.max_depth = max_depth
    
    def get_action(self, observation: Any, env: Any) -> Any:
        """使用BFS搜索找到最佳动作"""
        valid_actions = env.get_valid_actions()
        
        if not valid_actions:
            return None
        
        # 简化的搜索：评估每个动作的潜在价值
        best_action = None
        best_score = float('-inf')
        
        for action in valid_actions:
            score = self._evaluate_action_with_search(action, observation, env, self.max_depth)
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action
    
    def _evaluate_action_with_search(self, action: Any, observation: Any, env: Any, depth: int) -> float:
        """使用搜索评估动作价值"""
        if depth <= 0:
            return self._simple_evaluate(action, observation, env)
        
        # 模拟执行动作
        cloned_env = env.clone() if hasattr(env, 'clone') else env
        
        try:
            # 执行动作并获取新状态
            new_obs, reward, done, info = cloned_env.step(action)
            
            if done:
                return reward * 100  # 游戏结束的状态给高权重
            
            # 递归搜索对手的最佳回应
            opponent_actions = cloned_env.get_valid_actions()
            if not opponent_actions:
                return reward
            
            # 假设对手会选择对我们最不利的动作
            min_future_score = float('inf')
            for opp_action in opponent_actions[:3]:  # 限制搜索宽度
                future_score = self._evaluate_action_with_search(
                    opp_action, new_obs, cloned_env, depth - 1
                )
                min_future_score = min(min_future_score, future_score)
            
            return reward + 0.9 * min_future_score  # 折扣因子
            
        except:
            # 如果无法模拟，使用简单评估
            return self._simple_evaluate(action, observation, env)
    
    def _simple_evaluate(self, action: Any, observation: Any, env: Any) -> float:
        """简单的位置评估函数"""
        # 这里应该根据具体游戏实现评估逻辑
        # 例如：位置价值、安全性、距离目标等
        return random.random()  # 占位符


def test_simple_ais():
    """测试简单AI实现"""
    print("=== 测试简单AI实现 ===")
    
    from games.gomoku import GomokuEnv
    
    # 创建环境
    env = GomokuEnv(board_size=9, win_length=5)
    
    # 创建AI
    improved_random = ImprovedRandomBot("改进随机AI", 1)
    rule_based = RuleBasedGomokuBot("规则AI", 2)
    
    print(f"AI1: {improved_random.name}")
    print(f"AI2: {rule_based.name}")
    
    # 进行简单测试
    observation, info = env.reset()
    
    for i in range(10):
        current_player = env.game.current_player
        if current_player == 1:
            action = improved_random.get_action(observation, env)
        else:
            action = rule_based.get_action(observation, env)
        
        print(f"回合 {i+1}: 玩家 {current_player} 选择动作 {action}")
        
        if action is None:
            break
        
        observation, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            winner = env.get_winner()
            print(f"游戏结束，获胜者: {winner}")
            break
    
    print("测试完成！")


if __name__ == "__main__":
    test_simple_ais() 