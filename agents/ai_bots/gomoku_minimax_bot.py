from agents.base_agent import BaseAgent
import numpy as np
import time
import math
from typing import Tuple, List, Dict, Any

class GomokuMinimaxBot(BaseAgent):
    """
    专门为五子棋设计的Minimax AI
    """
    
    def __init__(self, name="GomokuMinimaxBot", player_id=1, max_depth=4, timeout=5.0):
        super().__init__(name, player_id)
        self.max_depth = max_depth
        self.timeout = timeout
        self.nodes_searched = 0
        self.start_time = 0
        self.transposition_table = {}
        
        # 五子棋评估模式定义
        self.patterns = {
            # 连五（胜利）
            'five': {'pattern': [1, 1, 1, 1, 1], 'score': 100000},
            
            # 活四（必胜）
            'live_four': {'pattern': [0, 1, 1, 1, 1, 0], 'score': 10000},
            
            # 冲四（需要堵）
            'rush_four': [
                {'pattern': [1, 1, 1, 1, 0], 'score': 1000},
                {'pattern': [0, 1, 1, 1, 1], 'score': 1000},
                {'pattern': [1, 0, 1, 1, 1], 'score': 1000},
                {'pattern': [1, 1, 0, 1, 1], 'score': 1000},
                {'pattern': [1, 1, 1, 0, 1], 'score': 1000},
            ],
            
            # 活三
            'live_three': [
                {'pattern': [0, 1, 1, 1, 0], 'score': 500},
                {'pattern': [0, 1, 0, 1, 1, 0], 'score': 500},
                {'pattern': [0, 1, 1, 0, 1, 0], 'score': 500},
            ],
            
            # 眠三
            'sleep_three': [
                {'pattern': [1, 1, 1, 0, 0], 'score': 50},
                {'pattern': [0, 0, 1, 1, 1], 'score': 50},
                {'pattern': [1, 0, 1, 1, 0], 'score': 50},
                {'pattern': [0, 1, 1, 0, 1], 'score': 50},
                {'pattern': [1, 0, 0, 1, 1], 'score': 50},
                {'pattern': [1, 1, 0, 0, 1], 'score': 50},
            ],
            
            # 活二
            'live_two': [
                {'pattern': [0, 1, 1, 0], 'score': 10},
                {'pattern': [0, 1, 0, 1, 0], 'score': 10},
            ],
            
            # 眠二
            'sleep_two': [
                {'pattern': [1, 1, 0, 0, 0], 'score': 2},
                {'pattern': [0, 0, 0, 1, 1], 'score': 2},
                {'pattern': [1, 0, 1, 0, 0], 'score': 2},
                {'pattern': [0, 0, 1, 0, 1], 'score': 2},
                {'pattern': [1, 0, 0, 1, 0], 'score': 2},
                {'pattern': [0, 1, 0, 0, 1], 'score': 2},
            ]
        }
    
    def get_action(self, observation, env):
        """获取最佳动作"""
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            return None

        if len(valid_actions) == 1:
            return valid_actions[0]

        # 检查是否是AI的第一步（只有当AI还没有下过子时才选择中心）
        ai_moves = 0
        for row in range(env.game.board_size):
            for col in range(env.game.board_size):
                if env.game.board[row, col] == self.player_id:
                    ai_moves += 1
        
        if ai_moves == 0:
            # AI的第一步，智能选择开局位置
            center = env.game.board_size // 2
            
            # 如果棋盘为空，选择中心
            if len(valid_actions) == env.game.board_size * env.game.board_size:
                return (center, center)
            
            # 如果对手已经下了子，选择附近的好位置
            best_opening_moves = []
            
            # 寻找对手棋子附近的好位置
            for row in range(env.game.board_size):
                for col in range(env.game.board_size):
                    if env.game.board[row, col] != 0 and env.game.board[row, col] != self.player_id:
                        # 在对手棋子附近寻找好位置
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0:
                                    continue
                                r, c = row + dr, col + dc
                                if (0 <= r < env.game.board_size and 0 <= c < env.game.board_size 
                                    and env.game.board[r, c] == 0 and (r, c) in valid_actions):
                                    best_opening_moves.append((r, c))
            
            # 如果找到了对手附近的位置，选择其中最好的
            if best_opening_moves:
                # 从中选择距离中心较近的位置
                best_move = min(best_opening_moves, 
                              key=lambda pos: abs(pos[0] - center) + abs(pos[1] - center))
                return best_move
            
            # 如果没有找到，选择中心附近的位置
            center_options = [(center, center), (center-1, center), (center+1, center), 
                            (center, center-1), (center, center+1)]
            for pos in center_options:
                if pos in valid_actions:
                    return pos
        
        self.start_time = time.time()
        self.nodes_searched = 0
        self.transposition_table.clear()
        
        # 智能动作排序：优先搜索有希望的位置
        sorted_actions = self._sort_actions(env.game, valid_actions)
        
        best_score = float('-inf')
        best_action = sorted_actions[0]
        
        # 迭代加深搜索
        for depth in range(1, self.max_depth + 1):
            if self._is_timeout():
                break
                
            current_best_score = float('-inf')
            current_best_action = sorted_actions[0]
            
            for action in sorted_actions:
                if self._is_timeout():
                    break
                
                try:
                    game_copy = env.game.clone()
                    game_copy.step(action)
                    
                    score = self.minimax_ab(game_copy, depth - 1, False, 
                                          float('-inf'), float('inf'))
                    
                    if score > current_best_score:
                        current_best_score = score
                        current_best_action = action
                        
                    # 如果找到必胜棋，立即返回
                    if score >= 100000:
                        print(f"GomokuBot found winning move: {action}")
                        return action
                        
                except Exception as e:
                    continue
            
            if not self._is_timeout():
                best_score = current_best_score
                best_action = current_best_action
                print(f"Depth {depth}: Best action {best_action} with score {best_score}")
        
        print(f"GomokuBot searched {self.nodes_searched} nodes in {time.time() - self.start_time:.3f}s, final choice: {best_action} (score: {best_score})")
        return best_action
    
    def minimax_ab(self, game, depth, maximizing_player, alpha, beta):
        """带Alpha-Beta剪枝的Minimax算法"""
        self.nodes_searched += 1
        
        if self._is_timeout():
            return self.evaluate_position(game)
        
        # 状态缓存
        state_key = self._get_state_hash(game)
        if state_key in self.transposition_table:
            cached = self.transposition_table[state_key]
            if cached['depth'] >= depth:
                return cached['score']
        
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
        
        # 动作排序优化
        sorted_actions = self._sort_actions(game, valid_actions)
        
        if maximizing_player:
            max_eval = float('-inf')
            for action in sorted_actions:
                if self._is_timeout():
                    break
                
                try:
                    game_copy = game.clone()
                    game_copy.step(action)
                    eval_score = self.minimax_ab(game_copy, depth - 1, False, alpha, beta)
                    max_eval = max(max_eval, eval_score)
                    alpha = max(alpha, eval_score)
                    
                    if beta <= alpha:  # Alpha-Beta剪枝
                        break
                except:
                    continue
            
            self.transposition_table[state_key] = {'score': max_eval, 'depth': depth}
            return max_eval
        else:
            min_eval = float('inf')
            for action in sorted_actions:
                if self._is_timeout():
                    break
                
                try:
                    game_copy = game.clone()
                    game_copy.step(action)
                    eval_score = self.minimax_ab(game_copy, depth - 1, True, alpha, beta)
                    min_eval = min(min_eval, eval_score)
                    beta = min(beta, eval_score)
                    
                    if beta <= alpha:  # Alpha-Beta剪枝
                        break
                except:
                    continue
            
            self.transposition_table[state_key] = {'score': min_eval, 'depth': depth}
            return min_eval
    
    def evaluate_position(self, game):
        """五子棋位置评估函数"""
        winner = game.get_winner()
        if winner == self.player_id:
            return 1000000
        elif winner is not None:
            return -1000000
        elif game.is_terminal():
            return 0
        
        my_score = self._evaluate_player(game.board, self.player_id)
        opponent_id = 3 - self.player_id  # 1->2, 2->1
        opponent_score = self._evaluate_player(game.board, opponent_id)
        
        # 大幅提高防守权重，特别是对活三和活四的防守
        defense_multiplier = 2.0
        return my_score - opponent_score * defense_multiplier
    
    def _evaluate_player(self, board, player):
        """评估某个玩家在棋盘上的得分"""
        score = 0
        board_size = board.shape[0]
        
        # 使用更精确的模式匹配评估
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for row in range(board_size):
            for col in range(board_size):
                for dr, dc in directions:
                    # 检查从当前位置开始的长线段
                    line = self._get_line(board, row, col, dr, dc, 9)
                    if len(line) >= 5:
                        line_score = self._evaluate_line(line, player)
                        score += line_score
        
        return score
    
    def _count_consecutive_length(self, board, start_row, start_col, dr, dc, player):
        """计算从起始位置开始的连续棋子长度"""
        board_size = board.shape[0]
        length = 0
        row, col = start_row, start_col
        
        # 向前计算
        while (0 <= row < board_size and 0 <= col < board_size and 
               board[row, col] == player):
            length += 1
            row += dr
            col += dc
        
        # 向后计算
        row, col = start_row - dr, start_col - dc
        while (0 <= row < board_size and 0 <= col < board_size and 
               board[row, col] == player):
            length += 1
            row -= dr
            col -= dc
        
        return length
    
    def _get_line(self, board, start_x, start_y, dx, dy, length):
        """获取从起始位置沿指定方向的线段"""
        line = []
        board_size = board.shape[0]
        
        for i in range(length):
            x = start_x + i * dx
            y = start_y + i * dy
            
            if 0 <= x < board_size and 0 <= y < board_size:
                line.append(board[x, y])
            else:
                line.append(-1)  # 边界外用-1表示
        
        return line
    
    def _evaluate_line(self, line, player):
        """评估线段中某个玩家的得分"""
        score = 0
        opponent = 3 - player
        
        # 将对手的棋子标记为-1，空位标记为0，自己的棋子标记为1
        normalized_line = []
        for cell in line:
            if cell == player:
                normalized_line.append(1)
            elif cell == opponent:
                normalized_line.append(-1)
            elif cell == 0:
                normalized_line.append(0)
            else:  # 边界
                normalized_line.append(-1)
        
        # 检查各种模式
        score += self._check_pattern(normalized_line, [1, 1, 1, 1, 1], 100000)  # 连五
        score += self._check_pattern(normalized_line, [0, 1, 1, 1, 1, 0], 10000)  # 活四
        
        # 冲四模式
        rush_four_patterns = [
            [1, 1, 1, 1, 0], [0, 1, 1, 1, 1], [1, 0, 1, 1, 1], 
            [1, 1, 0, 1, 1], [1, 1, 1, 0, 1]
        ]
        for pattern in rush_four_patterns:
            score += self._check_pattern(normalized_line, pattern, 1000)
        
        # 活三模式
        live_three_patterns = [
            [0, 1, 1, 1, 0], [0, 1, 0, 1, 1, 0], [0, 1, 1, 0, 1, 0]
        ]
        for pattern in live_three_patterns:
            score += self._check_pattern(normalized_line, pattern, 500)
        
        # 眠三模式
        sleep_three_patterns = [
            [1, 1, 1, 0, 0], [0, 0, 1, 1, 1], [1, 0, 1, 1, 0], 
            [0, 1, 1, 0, 1], [1, 0, 0, 1, 1], [1, 1, 0, 0, 1]
        ]
        for pattern in sleep_three_patterns:
            score += self._check_pattern(normalized_line, pattern, 50)
        
        # 活二模式
        live_two_patterns = [[0, 1, 1, 0], [0, 1, 0, 1, 0]]
        for pattern in live_two_patterns:
            score += self._check_pattern(normalized_line, pattern, 10)
        
        # 眠二模式
        sleep_two_patterns = [
            [1, 1, 0, 0, 0], [0, 0, 0, 1, 1], [1, 0, 1, 0, 0], 
            [0, 0, 1, 0, 1], [1, 0, 0, 1, 0], [0, 1, 0, 0, 1]
        ]
        for pattern in sleep_two_patterns:
            score += self._check_pattern(normalized_line, pattern, 2)
        
        return score
    
    def _check_pattern(self, line, pattern, score):
        """检查线段中是否包含指定模式"""
        count = 0
        pattern_len = len(pattern)
        line_len = len(line)
        
        for i in range(line_len - pattern_len + 1):
            if line[i:i + pattern_len] == pattern:
                count += 1
        
        return count * score
    
    def _sort_actions(self, game, actions):
        """根据启发式对动作进行排序，优先搜索有希望的位置"""
        if len(actions) <= 10:
            return actions
        
        scored_actions = []
        opponent_id = 3 - self.player_id
        
        for action in actions:
            row, col = action
            score = 0
            
            # 1. 检查是否能立即获胜
            temp_board = game.board.copy()
            temp_board[row, col] = self.player_id
            if self._check_win_at_position(temp_board, row, col, self.player_id):
                score += 100000
            
            # 2. 检查是否能阻止对手获胜 (非常重要!)
            temp_board = game.board.copy()
            temp_board[row, col] = opponent_id
            if self._check_win_at_position(temp_board, row, col, opponent_id):
                score += 90000  # 最高优先级：阻止对手获胜
            
            # 3. 检查是否能阻止对手形成活四或冲四或活三 (大幅提高权重!)
            if self._blocks_opponent_threat(game.board, row, col, opponent_id):
                score += 70000  # 非常高的权重：阻止威胁
                
                # 调试输出
                print(f"位置{action}可以阻挡对手威胁，得分+70000")
            
            # 4. 计算附近棋子的影响
            neighbor_count = 0
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    r, c = row + dr, col + dc
                    if (0 <= r < game.board_size and 0 <= c < game.board_size and 
                        game.board[r, c] != 0):
                        neighbor_count += 1
            score += neighbor_count * 1000
            
            # 5. 计算连线潜力
            temp_board = game.board.copy()
            temp_board[row, col] = self.player_id
            line_potential = self._calculate_line_potential(temp_board, row, col, self.player_id)
            score += line_potential * 100
            
            # 6. 中心区域加分（但权重较低）
            center = game.board_size // 2
            distance_to_center = abs(row - center) + abs(col - center)
            if neighbor_count == 0:
                score += max(0, 5 - distance_to_center)
            
            scored_actions.append((score, action))
        
        # 按得分降序排列
        scored_actions.sort(reverse=True)
        
        # 调试输出前几个候选动作
        print(f"前5个候选动作:")
        for i, (score, action) in enumerate(scored_actions[:5]):
            print(f"  {i+1}. {action}: {score}")
        
        return [action for score, action in scored_actions]
    
    def _check_win_at_position(self, board, row, col, player):
        """检查在指定位置下棋是否能获胜"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            length = self._count_consecutive_length(board, row, col, dr, dc, player)
            if length >= 5:
                return True
        return False
    
    def _blocks_opponent_threat(self, board, row, col, opponent_id):
        """检查在指定位置下棋是否能阻止对手的威胁"""
        # 简化的威胁检测：检查在这个位置放棋子是否能阻断对手的活三或活四
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            # 检查这个方向上对手的威胁
            if self._check_direction_threat(board, row, col, dr, dc, opponent_id):
                return True
        
        return False
    
    def _check_direction_threat(self, board, row, col, dr, dc, opponent_id):
        """检查特定方向上是否存在威胁"""
        board_size = board.shape[0]
        
        # 简化版本：检查这个位置是否在对手的活三线上
        # 检查通过这个位置的线段，看看是否包含活三模式
        
        # 扫描这个方向上更长的线段
        for line_len in [5, 6, 7]:
            for start_offset in range(-line_len+1, 1):
                start_r = row + start_offset * dr
                start_c = col + start_offset * dc
                
                if (0 <= start_r < board_size and 0 <= start_c < board_size and
                    0 <= start_r + (line_len-1)*dr < board_size and 
                    0 <= start_c + (line_len-1)*dc < board_size):
                    
                    # 获取线段
                    line = []
                    for i in range(line_len):
                        r = start_r + i * dr
                        c = start_c + i * dc
                        line.append(board[r, c])
                    
                    # 检查是否包含活三
                    if self._line_has_live_three(line, opponent_id, row - start_r, col - start_c, dr, dc):
                        return True
        
        return False
    
    def _line_has_live_three(self, line, player, target_row_offset, target_col_offset, dr, dc):
        """检查线段是否包含活三，并且目标位置能够阻断它"""
        # 计算目标位置在线段中的索引
        if dr == 0 and dc == 1:  # 水平
            target_idx = target_col_offset
        elif dr == 1 and dc == 0:  # 垂直
            target_idx = target_row_offset
        elif dr == 1 and dc == 1:  # 主对角线
            target_idx = target_row_offset
        elif dr == 1 and dc == -1:  # 副对角线
            target_idx = target_row_offset
        else:
            target_idx = -1
        
        if target_idx < 0 or target_idx >= len(line):
            return False
        
        # 标准化线段：对手棋子=1，空位=0，其他=-1
        normalized = []
        for cell in line:
            if cell == player:
                normalized.append(1)
            elif cell == 0:
                normalized.append(0)
            else:
                normalized.append(-1)
        
        # 检查标准活三模式 [0,1,1,1,0]
        for i in range(len(normalized) - 4):
            if normalized[i:i+5] == [0, 1, 1, 1, 0]:
                # 检查目标位置是否在活三的空位上
                if target_idx == i or target_idx == i + 4:
                    return True
        
        # 检查跳跃活三模式 [0,1,0,1,1,0]
        for i in range(len(normalized) - 5):
            if normalized[i:i+6] == [0, 1, 0, 1, 1, 0]:
                # 检查目标位置是否在关键空位上
                if target_idx == i or target_idx == i + 2 or target_idx == i + 5:
                    return True
        
        # 检查另一种跳跃活三 [0,1,1,0,1,0]
        for i in range(len(normalized) - 5):
            if normalized[i:i+6] == [0, 1, 1, 0, 1, 0]:
                if target_idx == i or target_idx == i + 3 or target_idx == i + 5:
                    return True
        
        return False
    
    def _calculate_line_potential(self, board, row, col, player):
        """计算在指定位置的连线潜力"""
        total_potential = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            length = self._count_consecutive_length(board, row, col, dr, dc, player)
            if length >= 2:
                total_potential += length * length  # 连线越长，潜力越大
        
        return total_potential
    
    def _get_state_hash(self, game):
        """生成游戏状态的哈希值"""
        try:
            board_str = ''.join(str(cell) for row in game.board for cell in row)
            return f"{board_str}_{game.current_player}"
        except:
            return str(time.time())
    
    def _is_timeout(self):
        """检查是否超时"""
        return time.time() - self.start_time > self.timeout
