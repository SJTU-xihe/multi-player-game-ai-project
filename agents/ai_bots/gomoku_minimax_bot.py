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
        
        # 五子棋评估模式定义 - 大幅调整权重确保活四绝对优先
        self.patterns = {
            # 连五（胜利）
            'five': {'pattern': [1, 1, 1, 1, 1], 'score': 100000},
            
            # 活四（必胜）- 大幅提高权重，确保绝对优先于冲四
            'live_four': {'pattern': [0, 1, 1, 1, 1, 0], 'score': 50000},
            
            # 冲四（需要堵）- 大幅降低权重，确保活四绝对优先
            'rush_four': [
                {'pattern': [1, 1, 1, 1, 0], 'score': 300},
                {'pattern': [0, 1, 1, 1, 1], 'score': 300},
                {'pattern': [1, 0, 1, 1, 1], 'score': 300},
                {'pattern': [1, 1, 0, 1, 1], 'score': 300},
                {'pattern': [1, 1, 1, 0, 1], 'score': 300},
            ],
            
            # 活三 - 进一步提高分值，因为活三能形成双威胁，但必须低于活四
            'live_three': [
                {'pattern': [0, 1, 1, 1, 0], 'score': 3000},  # 提高活三权重，但仍低于活四
                {'pattern': [0, 1, 0, 1, 1, 0], 'score': 3000},
                {'pattern': [0, 1, 1, 0, 1, 0], 'score': 3000},
            ],
            
            # 眠三 - 适度提高权重
            'sleep_three': [
                {'pattern': [1, 1, 1, 0, 0], 'score': 80},
                {'pattern': [0, 0, 1, 1, 1], 'score': 80},
                {'pattern': [1, 0, 1, 1, 0], 'score': 80},
                {'pattern': [0, 1, 1, 0, 1], 'score': 80},
                {'pattern': [1, 0, 0, 1, 1], 'score': 80},
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
        
        # 紧急威胁检测：如果有立即防守需求，强制执行
        urgent_defense = self._check_urgent_defense(env.game, valid_actions)
        if urgent_defense:
            print(f"检测到紧急威胁，强制防守: {urgent_defense}")
            return urgent_defense
        
        # 检查是否应该优先进攻
        should_attack = self._should_prioritize_attack(env.game)
        if should_attack:
            print("AI决定优先进攻!")
            attack_moves = self._get_attack_moves(env.game)
            if attack_moves:
                # 将进攻动作放在前面
                non_attack_moves = [a for a in sorted_actions if a not in attack_moves]
                sorted_actions = attack_moves + non_attack_moves
                print(f"找到{len(attack_moves)}个进攻位置: {attack_moves[:3]}...")
        
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
        
        # 检查进攻机会：如果我有活三或冲四，提高进攻权重
        my_threats = self._count_threats(game.board, self.player_id)
        opponent_threats = self._count_threats(game.board, opponent_id)
        
        # 优化的动态权重调整 - 大幅修改确保活四绝对优先
        if my_threats['live_four'] > 0:
            # 我方有活四，绝对优先进攻
            defense_multiplier = 0.5  # 进一步降低防守权重
            attack_bonus = my_threats['live_four'] * 20000  # 大幅提高活四奖励
            my_score += attack_bonus
        elif my_threats['live_three'] >= 2:
            # 我方有双活三，优先进攻
            defense_multiplier = 1.0
            attack_bonus = my_threats['live_three'] * 6000  # 提高双活三权重
            my_score += attack_bonus
        elif my_threats['live_three'] >= 1:
            # 我方有活三，适度降低防守权重
            if opponent_threats['live_three'] == 0:
                defense_multiplier = 1.2  # 对手没有活三时可以进攻
                attack_bonus = my_threats['live_three'] * 5000  # 提高单活三权重
            else:
                defense_multiplier = 1.6  # 对手也有活三时需要平衡
                attack_bonus = my_threats['live_three'] * 4000
            my_score += attack_bonus
        elif my_threats['rush_four'] >= 1:
            # 我方有冲四 - 大幅降低权重，优先寻找活四机会
            defense_multiplier = 1.8  # 提高防守权重
            attack_bonus = my_threats['rush_four'] * 800  # 大幅降低冲四奖励，从2000降到800
            my_score += attack_bonus
        elif opponent_threats['live_four'] > 0:
            # 对手有活四，必须防守
            defense_multiplier = 3.5
        elif opponent_threats['live_three'] >= 2:
            # 对手有双活三，高度警惕
            defense_multiplier = 3.0
        elif opponent_threats['live_three'] >= 1:
            # 对手有活三，提高防守权重
            defense_multiplier = 2.5
        elif opponent_threats['rush_four'] > 0:
            # 对手有冲四，需要防守
            defense_multiplier = 2.2
        else:
            # 一般情况下的防守权重
            defense_multiplier = 1.8
        
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
        
        # 检查各种模式 - 大幅调整权重确保活四绝对优先于冲四
        score += self._check_pattern(normalized_line, [1, 1, 1, 1, 1], 100000)  # 连五
        score += self._check_pattern(normalized_line, [0, 1, 1, 1, 1, 0], 50000)  # 活四 - 大幅提高权重
        
        # 冲四模式 - 大幅降低权重确保活四绝对优先
        rush_four_patterns = [
            [1, 1, 1, 1, 0], [0, 1, 1, 1, 1], [1, 0, 1, 1, 1], 
            [1, 1, 0, 1, 1], [1, 1, 1, 0, 1]
        ]
        for pattern in rush_four_patterns:
            score += self._check_pattern(normalized_line, pattern, 300)  # 从800大幅降低到300
        
        # 活三模式 - 保持较高权重，但确保低于活四
        live_three_patterns = [
            [0, 1, 1, 1, 0], [0, 1, 0, 1, 1, 0], [0, 1, 1, 0, 1, 0]
        ]
        for pattern in live_three_patterns:
            score += self._check_pattern(normalized_line, pattern, 3000)  # 提高活三权重，但仍低于活四
        
        # 眠三模式 - 适度提高权重
        sleep_three_patterns = [
            [1, 1, 1, 0, 0], [0, 0, 1, 1, 1], [1, 0, 1, 1, 0], 
            [0, 1, 1, 0, 1], [1, 0, 0, 1, 1], [1, 1, 0, 0, 1]
        ]
        for pattern in sleep_three_patterns:
            score += self._check_pattern(normalized_line, pattern, 100)  # 从80提高到100
        
        # 活二模式 - 适度提高权重
        live_two_patterns = [[0, 1, 1, 0], [0, 1, 0, 1, 0]]
        for pattern in live_two_patterns:
            score += self._check_pattern(normalized_line, pattern, 15)  # 从10提高到15
        
        # 眠二模式
        sleep_two_patterns = [
            [1, 1, 0, 0, 0], [0, 0, 0, 1, 1], [1, 0, 1, 0, 0], 
            [0, 0, 1, 0, 1], [1, 0, 0, 1, 0], [0, 1, 0, 0, 1]
        ]
        for pattern in sleep_two_patterns:
            score += self._check_pattern(normalized_line, pattern, 3)  # 从2提高到3
        
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
                score += 95000  # 最高优先级：阻止对手获胜
            
            # 3. 优先检查活四建立 - 新增优化逻辑
            temp_board = game.board.copy()
            temp_board[row, col] = self.player_id
            
            # 检查是否能形成活四（最高进攻优先级）
            can_form_live_four = False
            can_form_rush_four = False
            can_form_live_three = False
            
            directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
            for dr, dc in directions:
                line_length = self._count_consecutive_length(temp_board, row, col, dr, dc, self.player_id)
                
                if line_length >= 4:
                    if self._can_form_live_four(temp_board, row, col, dr, dc, self.player_id):
                        can_form_live_four = True
                        break
                    else:
                        can_form_rush_four = True
                elif line_length == 3:
                    if self._is_live_three(temp_board, row, col, dr, dc, self.player_id):
                        can_form_live_three = True
            
            # 根据威胁类型设置不同的分值 - 大幅调整确保活四绝对优先
            if can_form_live_four:
                score += 95000  # 活四获得最高优先级，接近胜利分数
                print(f"位置{action}形成活四，得分+95000")
            elif can_form_rush_four and not can_form_live_four:  # 只有在不能形成活四时才考虑冲四
                score += 15000  # 大幅降低冲四权重，远低于活四
                print(f"位置{action}形成冲四，得分+15000")
            elif can_form_live_three:
                score += 45000  # 活三重要，但必须低于活四
                print(f"位置{action}形成活三，得分+45000")
            
            # 4. 优化活三防守 - 检查能否阻止对手威胁
            opponent_live_three_threat = self._check_blocks_live_three_threat(game.board, row, col, opponent_id)
            # 检查对手紧急威胁（需要我们立即防守的）
            opponent_live_four_threat = self._check_blocks_live_four_threat(game.board, row, col, opponent_id) 
            opponent_rush_four_threat = self._check_blocks_rush_four_threat(game.board, row, col, opponent_id)
            
            # 防守权重优化
            if opponent_live_four_threat:
                score += 90000  # 阻止对手活四
                print(f"位置{action}阻止对手活四，得分+90000")
            elif opponent_rush_four_threat:
                score += 75000  # 阻止对手冲四
                print(f"位置{action}阻止对手冲四，得分+75000")
            elif opponent_live_three_threat:
                # 改进的活三防守逻辑：考虑先手优势
                my_current_threats = self._count_threats(game.board, self.player_id)
                if my_current_threats['live_three'] >= 1:
                    # 我方已有活三时，检查是否应该继续进攻而不是防守
                    if can_form_live_four:
                        # 如果能形成活四，直接进攻，不防守对手活三
                        print(f"位置{action}我方有活三且能形成活四，忽略对手活三威胁")
                        # 不给防守分数，让活四的85000分占主导
                    elif can_form_rush_four:
                        # 如果能形成冲四，也优先进攻
                        print(f"位置{action}我方有活三且能形成冲四，优先进攻")
                        # 不给防守分数
                    else:
                        # 我方有活三但此位置无法形成强威胁，适当防守
                        score += 30000  # 大幅降低防守权重，鼓励进攻
                        print(f"位置{action}阻止对手活三(我方有活三，权重降低)，得分+30000")
                else:
                    # 我方没有活三威胁时，必须防守对手活三
                    score += 95000  # 大幅提高权重，确保防守优先
                    print(f"位置{action}阻止对手活三(必须防守)，得分+95000")
            
            # 5. 智能进攻策略 - 形成活三后建立活四
            if can_form_live_three and not opponent_live_four_threat and not opponent_rush_four_threat:
                # 在形成活三且对手没有立即威胁时，检查下一步能否形成活四
                next_live_four_potential = self._check_next_live_four_potential(temp_board, self.player_id)
                if next_live_four_potential:
                    score += 25000  # 额外奖励有活四潜力的活三
                    print(f"位置{action}活三有活四潜力，额外得分+25000")
            
            # 6. 计算附近棋子的影响
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
            
            # 7. 计算连线潜力
            temp_board = game.board.copy()
            temp_board[row, col] = self.player_id
            line_potential = self._calculate_line_potential(temp_board, row, col, self.player_id)
            score += line_potential * 100
            
            # 8. 中心区域加分（但权重较低）
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
        
        # 标准化线段：对手棋子=1，空位=0，自己的棋子=-1
        normalized = []
        for cell in line:
            if cell == player:
                normalized.append(-1)
            elif cell == 0:
                normalized.append(0)
            else:
                normalized.append(1)
        
        # 检查标准活三模式 [0,1,1,1,0]
        for i in range(len(normalized) - 4):
            if normalized[i:i+5] == [0, -1, -1, -1, 0]:
                # 检查目标位置是否在活三的空位上
                if target_idx == i or target_idx == i + 4:
                    return True
        
        # 检查跳跃活三模式 [0,1,0,1,1,0]
        for i in range(len(normalized) - 5):
            if normalized[i:i+6] == [0, -1, 0, -1, -1, 0]:
                # 检查目标位置是否在关键空位上
                if target_idx == i or target_idx == i + 2 or target_idx == i + 5:
                    return True
        
        # 检查另一种跳跃活三 [0,1,1,0,1,0]
        for i in range(len(normalized) - 5):
            if normalized[i:i+6] == [0, -1, -1, 0, -1, 0]:
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
    
    def _count_threats(self, board, player):
        """计算玩家在棋盘上的威胁数量"""
        threats = {
            'five': 0,
            'live_four': 0,
            'rush_four': 0,
            'live_three': 0,
            'sleep_three': 0,
            'live_two': 0
        }
        
        board_size = board.shape[0]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        # 用集合避免重复计数
        counted_patterns = set()
        
        # 检查每个方向的威胁
        for row in range(board_size):
            for col in range(board_size):
                for dr, dc in directions:
                    # 检查从当前位置开始的长线段
                    line = self._get_line(board, row, col, dr, dc, 9)
                    if len(line) >= 5:
                        line_threats = self._count_line_threats_precise(line, player, row, col, dr, dc)
                        for threat_type, positions in line_threats.items():
                            for pos in positions:
                                pattern_key = (threat_type, pos, dr, dc)
                                if pattern_key not in counted_patterns:
                                    threats[threat_type] += 1
                                    counted_patterns.add(pattern_key)
        
        return threats
    
    def _count_line_threats_precise(self, line, player, start_row, start_col, dr, dc):
        """精确计算线段中的威胁数量，返回威胁位置"""
        threats = {
            'five': [],
            'live_four': [],
            'rush_four': [],
            'live_three': [],
            'sleep_three': [],
            'live_two': []
        }
        
        opponent = 3 - player
        
        # 标准化线段
        normalized_line = []
        for cell in line:
            if cell == player:
                normalized_line.append(1)
            elif cell == opponent:
                normalized_line.append(-1)
            elif cell == 0:
                normalized_line.append(0)
            else:
                normalized_line.append(-1)
        
        # 检查各种威胁模式，并记录位置
        self._find_pattern_positions(normalized_line, [1, 1, 1, 1, 1], threats['five'], start_row, start_col, dr, dc)
        self._find_pattern_positions(normalized_line, [0, 1, 1, 1, 1, 0], threats['live_four'], start_row, start_col, dr, dc)
        
        # 冲四模式
        rush_four_patterns = [
            [1, 1, 1, 1, 0], [0, 1, 1, 1, 1], [1, 0, 1, 1, 1], 
            [1, 1, 0, 1, 1], [1, 1, 1, 0, 1]
        ]
        for pattern in rush_four_patterns:
            self._find_pattern_positions(normalized_line, pattern, threats['rush_four'], start_row, start_col, dr, dc)
        
        # 活三模式
        live_three_patterns = [
            [0, 1, 1, 1, 0], [0, 1, 0, 1, 1, 0], [0, 1, 1, 0, 1, 0]
        ]
        for pattern in live_three_patterns:
            self._find_pattern_positions(normalized_line, pattern, threats['live_three'], start_row, start_col, dr, dc)
        
        return threats
    
    def _find_pattern_positions(self, line, pattern, result_list, start_row, start_col, dr, dc):
        """在线段中查找模式并记录位置"""
        pattern_len = len(pattern)
        line_len = len(line)
        
        for i in range(line_len - pattern_len + 1):
            if line[i:i + pattern_len] == pattern:
                # 计算模式在棋盘上的实际位置
                pattern_row = start_row + i * dr
                pattern_col = start_col + i * dc
                result_list.append((pattern_row, pattern_col))
    
    def _count_pattern_in_line(self, line, pattern):
        """计算线段中特定模式的出现次数"""
        count = 0
        pattern_len = len(pattern)
        line_len = len(line)
        
        for i in range(line_len - pattern_len + 1):
            if line[i:i + pattern_len] == pattern:
                count += 1
        
        return count
    
    def _should_prioritize_attack(self, game):
        """判断是否应该优先进攻 - 重点优化活四判断"""
        my_threats = self._count_threats(game.board, self.player_id)
        opponent_threats = self._count_threats(game.board, 3 - self.player_id)
        
        # 如果我有活四，绝对优先进攻
        if my_threats['live_four'] > 0:
            print("发现活四机会，绝对优先进攻！")
            return True
        
        # 如果我有活三，优先考虑进攻
        if my_threats['live_three'] >= 1:
            # 除非对手有立即的致命威胁
            if opponent_threats['live_four'] == 0:
                print(f"我方有{my_threats['live_three']}个活三，优先进攻")
                return True
        
        # 如果我有双活三或活三+冲四组合，绝对优先进攻
        if (my_threats['live_three'] >= 2 or 
            (my_threats['live_three'] >= 1 and my_threats['rush_four'] >= 1)):
            print("发现双威胁组合，绝对优先进攻！")
            return True
        
        # 冲四的优先级大幅降低，只有在没有活四和活三机会时才考虑
        if my_threats['rush_four'] >= 1 and my_threats['live_three'] == 0:
            # 检查是否可能通过冲四转换为活四
            if self._can_rush_four_become_live_four(game):
                print("冲四有转化为活四的潜力，考虑进攻")
                return True
            elif opponent_threats['live_four'] == 0 and opponent_threats['live_three'] <= 1:
                print("对手威胁较小且我方只有冲四，适度进攻")
                return True
        
        return False

    def _can_rush_four_become_live_four(self, game):
        """检查冲四是否能在下一步转化为活四"""
        board_size = game.board_size
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        # 找到所有空位
        for row in range(board_size):
            for col in range(board_size):
                if game.board[row, col] == 0:
                    # 检查在这个位置下棋是否能让现有冲四转化为活四
                    temp_board = game.board.copy()
                    temp_board[row, col] = self.player_id
                    
                    # 检查这个位置是否能形成活四
                    for dr, dc in directions:
                        if self._can_form_live_four(temp_board, row, col, dr, dc, self.player_id):
                            # 进一步检查：这个活四是否是从冲四转化来的
                            original_length = self._count_consecutive_length(game.board, row, col, dr, dc, self.player_id)
                            if original_length >= 3:  # 原来就有连子基础
                                return True
        
        return False

    def _get_attack_moves(self, game):
        """获取所有可能的进攻位置 - 优先活四"""
        attack_moves = []
        live_four_moves = []  # 专门收集活四位置
        live_three_moves = []  # 专门收集活三位置
        rush_four_moves = []  # 冲四位置权重最低
        
        valid_actions = game.get_valid_actions()
        
        for action in valid_actions:
            row, col = action
            temp_board = game.board.copy()
            temp_board[row, col] = self.player_id
            
            # 检查这个位置能形成的威胁类型
            can_form_live_four = False
            can_form_live_three = False
            can_form_rush_four = False
            
            directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
            for dr, dc in directions:
                line_length = self._count_consecutive_length(temp_board, row, col, dr, dc, self.player_id)
                
                if line_length >= 4:
                    if self._can_form_live_four(temp_board, row, col, dr, dc, self.player_id):
                        can_form_live_four = True
                        break  # 活四优先级最高，找到就退出
                    else:
                        can_form_rush_four = True
                elif line_length == 3:
                    if self._is_live_three(temp_board, row, col, dr, dc, self.player_id):
                        can_form_live_three = True
            
            # 根据威胁类型分类存储
            if can_form_live_four:
                live_four_moves.append(action)
                print(f"发现活四机会: {action}")
            elif can_form_live_three:
                live_three_moves.append(action)
            elif can_form_rush_four:
                rush_four_moves.append(action)
            else:
                # 检查是否能形成新的威胁
                threats_after = self._count_threats(temp_board, self.player_id)
                threats_before = self._count_threats(game.board, self.player_id)
                
                if (threats_after['live_three'] > threats_before['live_three'] or
                    threats_after['rush_four'] > threats_before['rush_four'] or
                    threats_after['live_four'] > threats_before['live_four']):
                    attack_moves.append(action)
        
        # 按优先级排序：活四 > 活三 > 其他进攻 > 冲四
        result = live_four_moves + live_three_moves + attack_moves + rush_four_moves
        
        if live_four_moves:
            print(f"找到{len(live_four_moves)}个活四机会，优先考虑！")
        elif live_three_moves:
            print(f"找到{len(live_three_moves)}个活三机会")
        elif rush_four_moves:
            print(f"只找到{len(rush_four_moves)}个冲四机会")
            
        return result
    
    def _is_live_three(self, board, row, col, dr, dc, player):
        """检查指定位置和方向是否形成活三"""
        board_size = board.shape[0]
        
        # 找到连续棋子的范围
        start_r, start_c = row, col
        end_r, end_c = row, col
        
        # 向前扩展
        while (0 <= start_r - dr < board_size and 0 <= start_c - dc < board_size and
               board[start_r - dr, start_c - dc] == player):
            start_r -= dr
            start_c -= dc
        
        # 向后扩展
        while (0 <= end_r + dr < board_size and 0 <= end_c + dc < board_size and
               board[end_r + dr, end_c + dc] == player):
            end_r += dr
            end_c += dc
        
        # 计算连续长度
        length = max(abs(end_r - start_r), abs(end_c - start_c)) + 1
        
        # 检查连续长度是否为3，且两端都是空位
        if length == 3:
            front_empty = (0 <= start_r - dr < board_size and 0 <= start_c - dc < board_size and
                          board[start_r - dr, start_c - dc] == 0)
            back_empty = (0 <= end_r + dr < board_size and 0 <= end_c + dc < board_size and
                         board[end_r + dr, end_c + dc] == 0)
            
            return front_empty and back_empty
        
        return False
    
    def _can_form_live_four(self, board, row, col, dr, dc, player):
        """检查指定位置和方向是否能形成活四"""
        board_size = board.shape[0]
        
        # 找到连续棋子的范围
        start_r, start_c = row, col
        end_r, end_c = row, col
        
        # 向前扩展
        while (0 <= start_r - dr < board_size and 0 <= start_c - dc < board_size and
               board[start_r - dr, start_c - dc] == player):
            start_r -= dr
            start_c -= dc
        
        # 向后扩展
        while (0 <= end_r + dr < board_size and 0 <= end_c + dc < board_size and
               board[end_r + dr, end_c + dc] == player):
            end_r += dr
            end_c += dc
        
        # 计算连续长度
        length = max(abs(end_r - start_r), abs(end_c - start_c)) + 1
        
        # 检查连续长度是否为4，且两端都是空位
        if length == 4:
            front_empty = (0 <= start_r - dr < board_size and 0 <= start_c - dc < board_size and
                          board[start_r - dr, start_c - dc] == 0)
            back_empty = (0 <= end_r + dr < board_size and 0 <= end_c + dc < board_size and
                         board[end_r + dr, end_c + dc] == 0)
            return front_empty and back_empty
        
        return False

    def _check_enhanced_live_three_threat(self, board, row, col, player):
        """增强的活三威胁检测"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            # 检查在这个位置放置对手棋子是否能形成活三
            temp_board = board.copy()
            temp_board[row, col] = player
            
            # 检查这个方向是否能形成活三
            if self._is_live_three(temp_board, row, col, dr, dc, player):
                return True
                
            # 检查这个位置是否能完成对手已有的活三结构
            board_size = board.shape[0]
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
                        
                        # 模拟在目标位置放置对手棋子
                        target_idx = (row - start_r) if dr != 0 else (col - start_c)
                        if 0 <= target_idx < len(line):
                            test_line = line.copy()
                            test_line[target_idx] = player
                            
                            # 检查是否形成活三模式
                            if self._has_live_three_pattern(test_line, player):
                                return True
        
        return False
    
    def _check_live_four_threat(self, board, row, col, player):
        """检查活四威胁"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            temp_board = board.copy()
            temp_board[row, col] = player
            
            if self._can_form_live_four(temp_board, row, col, dr, dc, player):
                return True
        
        return False
    
    def _check_rush_four_threat(self, board, row, col, player):
        """检查冲四威胁"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            temp_board = board.copy()
            temp_board[row, col] = player
            
            line_length = self._count_consecutive_length(temp_board, row, col, dr, dc, player)
            if line_length >= 4 and not self._can_form_live_four(temp_board, row, col, dr, dc, player):
                return True
        
        return False
    
    def _check_gap_live_three(self, board, row, col, dr, dc, player):
        """检查断点活三模式（如 .XX.X.）"""
        board_size = board.shape[0]
        
        # 检查以当前位置为中心的7格范围内的断点活三模式
        for start_offset in range(-3, 1):
            positions = []
            for i in range(7):
                r = row + (start_offset + i) * dr
                c = col + (start_offset + i) * dc
                if 0 <= r < board_size and 0 <= c < board_size:
                    positions.append((r, c, board[r, c]))
                else:
                    positions.append((r, c, -1))  # 边界标记
            
            if len(positions) == 7:
                # 提取值序列
                values = [pos[2] for pos in positions]
                
                # 检查各种断点活三模式
                patterns = [
                    [0, player, player, 0, player, 0, 0],  # .XX.X..
                    [0, 0, player, player, 0, player, 0],  # ..XX.X.
                    [0, player, 0, player, player, 0, 0],  # .X.XX..
                    [0, 0, player, 0, player, player, 0],  # ..X.XX.
                ]
                
                for pattern in patterns:
                    if values == pattern and (row, col) in [(positions[i][0], positions[i][1]) for i in range(7) if pattern[i] == 0]:
                        return True
        
        return False
    
    def _check_next_live_four_potential(self, board, player):
        """检查下一步是否有活四潜力"""
        board_size = board.shape[0]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        # 找到所有空位
        for row in range(board_size):
            for col in range(board_size):
                if board[row, col] == 0:
                    # 检查在这个位置下棋是否能形成活四
                    temp_board = board.copy()
                    temp_board[row, col] = player
                    
                    for dr, dc in directions:
                        if self._can_form_live_four(temp_board, row, col, dr, dc, player):
                            return True
        
        return False

    def _line_has_opponent_live_three(self, line, player, target_idx):
        """检查线段是否包含对手的活三，并且目标位置能够阻断它"""
        if target_idx < 0 or target_idx >= len(line):
            return False
        
        # 标准化线段：对手棋子=1，空位=0，我方棋子=-1
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
        
        return False

    def _has_live_three_pattern(self, line, player):
        """检查线段中是否包含活三模式"""
        opponent = 3 - player
        
        # 标准化线段
        normalized = []
        for cell in line:
            if cell == player:
                normalized.append(1)
            elif cell == opponent:
                normalized.append(-1)
            elif cell == 0:
                normalized.append(0)
            else:
                normalized.append(-1)  # 边界
        
        # 检查活三模式 [0,1,1,1,0]
        for i in range(len(normalized) - 4):
            if normalized[i:i+5] == [0, 1, 1, 1, 0]:
                return True
        
        # 检查跳跃活三模式
        for i in range(len(normalized) - 5):
            if (normalized[i:i+6] == [0, 1, 0, 1, 1, 0] or 
                normalized[i:i+6] == [0, 1, 1, 0, 1, 0]):
                return True
        
        return False

    def _check_blocks_live_three_threat(self, board, row, col, opponent_id):
        """检查在指定位置放棋是否能阻止对手的活三威胁"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        # 暂时在目标位置放上我方棋子
        temp_board = board.copy()
        temp_board[row, col] = 3 - opponent_id  # 我方ID
        
        # 检查附近是否有对手的活三被这步棋破坏了
        board_size = board.shape[0]
        for dr, dc in directions:
            # 检查这个方向上是否有对手活三被阻断
            for offset in range(-4, 5):  # 检查附近位置
                check_r = row + offset * dr
                check_c = col + offset * dc
                
                if (0 <= check_r < board_size and 0 <= check_c < board_size and
                    board[check_r, check_c] == opponent_id):
                    
                    # 检查原来的棋盘上这里是否有活三
                    if self._is_live_three(board, check_r, check_c, dr, dc, opponent_id):
                        # 检查放棋后是否还有活三
                        if not self._is_live_three(temp_board, check_r, check_c, dr, dc, opponent_id):
                            return True
        
        return False

    def _check_blocks_live_four_threat(self, board, row, col, opponent_id):
        """检查在指定位置放棋是否能阻止对手的活四威胁"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        # 暂时在目标位置放上我方棋子
        temp_board = board.copy()
        temp_board[row, col] = 3 - opponent_id  # 我方ID
        
        # 检查附近是否有对手的活四被这步棋破坏了
        board_size = board.shape[0]
        for dr, dc in directions:
            # 检查这个方向上是否有对手活四被阻断
            for offset in range(-4, 5):  # 检查附近位置
                check_r = row + offset * dr
                check_c = col + offset * dc
                
                if (0 <= check_r < board_size and 0 <= check_c < board_size and
                    board[check_r, check_c] == opponent_id):
                    
                    # 检查原来的棋盘上这里是否有活四
                    if self._can_form_live_four(board, check_r, check_c, dr, dc, opponent_id):
                        # 检查放棋后是否还有活四
                        if not self._can_form_live_four(temp_board, check_r, check_c, dr, dc, opponent_id):
                            return True
        
        return False

    def _check_blocks_rush_four_threat(self, board, row, col, opponent_id):
        """检查在指定位置放棋是否能阻止对手的冲四威胁"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        # 暂时在目标位置放上我方棋子
        temp_board = board.copy()
        temp_board[row, col] = 3 - opponent_id  # 我方ID
        
        # 检查附近是否有对手的冲四被这步棋破坏了
        board_size = board.shape[0]
        for dr, dc in directions:
            # 检查这个方向上是否有对手冲四被阻断
            for offset in range(-4, 5):  # 检查附近位置
                check_r = row + offset * dr
                check_c = col + offset * dc
                
                if (0 <= check_r < board_size and 0 <= check_c < board_size and
                    board[check_r, check_c] == opponent_id):
                    
                    # 检查原来的棋盘上这里是否能形成冲四
                    line_length = self._count_consecutive_length(board, check_r, check_c, dr, dc, opponent_id)
                    if (line_length >= 4 and not self._can_form_live_four(board, check_r, check_c, dr, dc, opponent_id)):
                        # 检查放棋后是否还能形成冲四
                        new_line_length = self._count_consecutive_length(temp_board, check_r, check_c, dr, dc, opponent_id)
                        if (new_line_length < 4):
                            return True
        
        return False

    def _check_urgent_defense(self, game, valid_actions):
        """检查是否有紧急防守需求（对手活三/活四威胁）"""
        opponent_id = 3 - self.player_id
        
        # 检查所有位置，找到能阻止对手致命威胁的位置
        urgent_positions = []
        
        for action in valid_actions:
            row, col = action
            
            # 检查是否能阻止对手活四
            if self._check_blocks_live_four_threat(game.board, row, col, opponent_id):
                print(f"位置{action}可阻止对手活四威胁")
                return action  # 活四威胁是最紧急的，立即返回
            
            # 检查是否能阻止对手活三
            if self._check_blocks_live_three_threat(game.board, row, col, opponent_id):
                urgent_positions.append(action)
        
        # 如果有活三威胁需要防守
        if urgent_positions:
            print(f"发现活三威胁防守位置: {urgent_positions}")
            
            # 检查我方是否已有活三优势
            my_threats = self._count_threats(game.board, self.player_id)
            if my_threats['live_three'] >= 1:
                print(f"我方已有{my_threats['live_three']}个活三，不强制防守对手活三，交由正常评估")
                return None  # 不强制防守，让正常的动作排序和评估来决定
            
            # 我方没有活三时，才强制防守
            print("我方无活三优势，强制防守对手活三")
            sorted_actions = self._sort_actions(game, urgent_positions)
            return sorted_actions[0]
        
        return None
