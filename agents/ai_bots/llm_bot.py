"""
大语言模型AI Bot
实现基于LLM的智能决策系统
"""

import json
import random
import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from agents.base_agent import BaseAgent


class LLMBot(BaseAgent):
    """大语言模型AI Bot"""
    
    def __init__(self, name: str = "LLM Bot", player_id: int = 1, 
                 model_name: str = 'gpt-3.5-turbo', api_key: Optional[str] = None,
                 use_local_simulation: bool = True, **kwargs):
        super().__init__(name, player_id)
        self.model_name = model_name
        self.api_key = api_key
        self.use_local_simulation = use_local_simulation
        self.reasoning_depth = kwargs.get('reasoning_depth', 3)
        self.temperature = kwargs.get('temperature', 0.7)
        
    def get_action(self, observation: Dict[str, Any], env) -> Optional[str]:
        """获取动作"""
        try:
            # 将游戏状态转换为文本描述
            game_description = self.observation_to_text(observation, env)
            
            # 获取有效动作
            valid_actions = self._get_valid_actions(observation, env)
            if not valid_actions:
                return None
            
            # 构建提示词
            prompt = self._build_prompt(game_description, valid_actions, env)
            
            # 调用LLM获取决策
            if self.use_local_simulation or not self.api_key:
                response = self._simulate_llm_reasoning(prompt, observation, env)
            else:
                response = self.call_llm(prompt)
            
            # 解析动作
            action = self.parse_action_from_response(response, valid_actions)
            
            return action if action else random.choice(valid_actions)
            
        except Exception as e:
            print(f"LLM Bot error: {e}")
            valid_actions = self._get_valid_actions(observation, env)
            return random.choice(valid_actions) if valid_actions else None
    
    def observation_to_text(self, observation: Dict[str, Any], env) -> str:
        """将游戏状态转换为文本描述"""
        game_type = env.__class__.__name__.lower()
        
        if 'sokoban' in game_type:
            return self._sokoban_to_text(observation)
        elif 'gomoku' in game_type:
            return self._gomoku_to_text(observation)
        elif 'snake' in game_type:
            return self._snake_to_text(observation)
        else:
            return self._generic_to_text(observation)
    
    def _sokoban_to_text(self, observation: Dict[str, Any]) -> str:
        """推箱子游戏状态转文本"""
        board = observation['board']
        player_pos = None
        
        # 确定玩家位置
        if self.player_id == 1:
            player_pos = tuple(observation['player1_pos']) if observation['player1_pos'][0] >= 0 else None
        else:
            player_pos = tuple(observation['player2_pos']) if observation['player2_pos'][0] >= 0 else None
        
        # 分析棋盘状态
        boxes = []
        targets = []
        walls = []
        
        symbol_map = {
            0: '空地',
            1: '墙壁',
            2: '目标点',
            3: '箱子',
            4: '箱子在目标上',
            5: '玩家1',
            6: '玩家2',
            7: '玩家1在目标上',
            8: '玩家2在目标上'
        }
        
        for row in range(board.shape[0]):
            for col in range(board.shape[1]):
                cell = board[row, col]
                if cell == 1:
                    walls.append((row, col))
                elif cell in [3, 4]:
                    boxes.append((row, col))
                    if cell == 4:
                        targets.append((row, col))
                elif cell in [2, 7, 8]:
                    targets.append((row, col))
        
        boxes_on_targets = len([box for box in boxes if box in targets])
        total_targets = len(set(targets))
        
        description = f"""
推箱子游戏状态分析：
- 地图大小：{board.shape[0]}x{board.shape[1]}
- 玩家{self.player_id}位置：{player_pos}
- 箱子总数：{len(boxes)}
- 目标点总数：{total_targets}
- 已完成箱子：{boxes_on_targets}
- 完成度：{boxes_on_targets/max(1,total_targets)*100:.1f}%

当前状态：
- 箱子位置：{boxes}
- 目标点位置：{set(targets)}
- 已完成的箱子：{[box for box in boxes if box in targets]}
- 未完成的箱子：{[box for box in boxes if box not in targets]}

战略分析：
- 需要移动的箱子数：{len(boxes) - boxes_on_targets}
- 玩家距离最近箱子：{self._calculate_min_distance(player_pos, [box for box in boxes if box not in targets]) if player_pos and boxes else '无'}
"""
        return description
    
    def _gomoku_to_text(self, observation: Dict[str, Any]) -> str:
        """五子棋游戏状态转文本"""
        board = observation.get('board', np.zeros((15, 15)))
        
        # 统计棋子
        player1_stones = np.sum(board == 1)
        player2_stones = np.sum(board == 2)
        
        # 分析威胁和机会
        threats = self._analyze_gomoku_threats(board, self.player_id)
        opportunities = self._analyze_gomoku_opportunities(board, self.player_id)
        
        description = f"""
五子棋游戏状态分析：
- 棋盘：15x15
- 玩家1棋子数：{player1_stones}
- 玩家2棋子数：{player2_stones}
- 当前轮到：玩家{self.player_id}

威胁分析：
- 对手即将获胜的位置：{threats}
- 我方的获胜机会：{opportunities}

策略建议：
- 优先级1：阻止对手获胜
- 优先级2：寻找自己的获胜机会
- 优先级3：构建攻击和防守阵型
"""
        return description
    
    def _snake_to_text(self, observation: Dict[str, Any]) -> str:
        """贪吃蛇游戏状态转文本"""
        # 基础的贪吃蛇状态分析
        description = f"""
贪吃蛇游戏状态：
- 当前玩家：{self.player_id}
- 游戏进行中
- 需要避开墙壁和自己的身体
- 寻找食物并安全移动
"""
        return description
    
    def _generic_to_text(self, observation: Dict[str, Any]) -> str:
        """通用游戏状态转文本"""
        return f"""
 游戏状态：
- 当前玩家：{self.player_id}
- 观察信息：{str(observation)[:200]}...
"""
    
    def _build_prompt(self, game_description: str, valid_actions: List[str], env) -> str:
        """构建LLM提示词"""
        game_type = env.__class__.__name__
        
        prompt = f"""
你是一个专业的{game_type}游戏AI。请分析当前局势并选择最佳动作。

{game_description}

可选动作：{valid_actions}

请按以下步骤分析：
1. 评估当前局势
2. 识别关键威胁和机会
3. 考虑每个可选动作的后果
4. 选择最优动作

最终请只返回一个动作，格式为：ACTION: [动作名称]

例如：ACTION: UP
"""
        return prompt
    
    def _simulate_llm_reasoning(self, prompt: str, observation: Dict[str, Any], env) -> str:
        """模拟LLM推理过程（本地实现）"""
        game_type = env.__class__.__name__.lower()
        
        if 'sokoban' in game_type:
            return self._sokoban_reasoning(observation, env)
        elif 'gomoku' in game_type:
            return self._gomoku_reasoning(observation, env)
        else:
            return self._default_reasoning(observation, env)
    
    def _sokoban_reasoning(self, observation: Dict[str, Any], env) -> str:
        """推箱子游戏推理 - 增强推箱子意识"""
        board = observation['board']
        valid_actions = self._get_valid_actions(observation, env)
        
        # 获取玩家位置
        if self.player_id == 1:
            player_pos = tuple(observation['player1_pos']) if observation['player1_pos'][0] >= 0 else None
        else:
            player_pos = tuple(observation['player2_pos']) if observation['player2_pos'][0] >= 0 else None
        
        if not player_pos:
            return f"ACTION: {random.choice(valid_actions)}"
        
        # 分析箱子和目标
        boxes = []
        targets = []
        
        for row in range(board.shape[0]):
            for col in range(board.shape[1]):
                cell = board[row, col]
                if cell in [3, 4]:  # 箱子
                    boxes.append((row, col))
                if cell in [2, 4, 7, 8]:  # 目标点
                    targets.append((row, col))
        
        # 计算完成情况
        completed_boxes = [box for box in boxes if box in targets]
        incomplete_boxes = [box for box in boxes if box not in targets]
        completion_ratio = len(completed_boxes) / max(1, len(targets))
        
        # 如果游戏接近完成，更加积极
        urgency_factor = 1.0
        if completion_ratio > 0.5:
            urgency_factor = 2.0
        elif completion_ratio > 0.8:
            urgency_factor = 3.0
        
        # 高级策略分析系统
        action_strategies = self._analyze_sokoban_strategies(
            player_pos, boxes, targets, incomplete_boxes, board, valid_actions, urgency_factor
        )
        
        # 选择最佳策略和动作
        best_strategy = max(action_strategies, key=lambda x: x['priority_score'])
        best_action = best_strategy['action']
        
        # 生成详细推理解释
        reasoning = f"""
🧠 高级推箱子战略分析：
========================================
📍 玩家位置：{player_pos}
📦 总箱子数：{len(boxes)} | 已完成：{len(completed_boxes)}/{len(targets)} ({completion_ratio:.1%})
🎯 未完成箱子：{incomplete_boxes}
⚡ 紧迫程度：{urgency_factor:.1f}x

🎯 策略评估排序：
{chr(10).join([f"  {i+1}. {s['strategy_name']}: {s['priority_score']:.1f}分 -> {s['action']}" 
               for i, s in enumerate(sorted(action_strategies, key=lambda x: x['priority_score'], reverse=True))])}

🏆 最优策略：{best_strategy['strategy_name']}
📋 策略描述：{best_strategy['reasoning']}
🎲 执行动作：{best_action}

ACTION: {best_action}
"""
        return reasoning
    
    def _analyze_sokoban_strategies(self, player_pos, boxes, targets, incomplete_boxes, board, valid_actions, urgency_factor):
        """分析推箱子的各种策略"""
        strategies = []
        
        directions = {
            'UP': (-1, 0),
            'DOWN': (1, 0),
            'LEFT': (0, -1),
            'RIGHT': (0, 1)
        }
        
        for action in valid_actions:
            if action not in directions:
                continue
                
            dr, dc = directions[action]
            target_pos = (player_pos[0] + dr, player_pos[1] + dc)
            
            # 策略1：直接完成箱子（最高优先级）
            if target_pos in boxes:
                box_new_pos = (target_pos[0] + dr, target_pos[1] + dc)
                if self._is_valid_push_position_llm(box_new_pos, board, boxes):
                    if box_new_pos in targets:
                        strategies.append({
                            'action': action,
                            'strategy_name': '🎯 完美击杀',
                            'priority_score': 10000 * urgency_factor,
                            'reasoning': f'直接推箱子({target_pos})到目标({box_new_pos})，游戏进展+1！'
                        })
                    else:
                        # 检查是否让箱子更接近目标
                        improvement = self._calculate_target_improvement_llm(target_pos, box_new_pos, targets, boxes)
                        if improvement > 0:
                            strategies.append({
                                'action': action,
                                'strategy_name': '📦 战术推进',
                                'priority_score': 1000 + improvement * 500 * urgency_factor,
                                'reasoning': f'推箱子({target_pos})→({box_new_pos})，接近目标{improvement:.1f}格'
                            })
                        else:
                            strategies.append({
                                'action': action,
                                'strategy_name': '🚧 谨慎推动',
                                'priority_score': 100,
                                'reasoning': f'推箱子但效果未知，需要谨慎评估'
                            })
            
            # 策略2：接近有价值的箱子
            elif incomplete_boxes:
                approach_value = self._calculate_approach_value_llm(target_pos, incomplete_boxes, targets, urgency_factor)
                if approach_value > 0:
                    strategies.append({
                        'action': action,
                        'strategy_name': '🚶 战略定位',
                        'priority_score': approach_value,
                        'reasoning': f'移动到({target_pos})，接近有价值的箱子位置'
                    })
            
            # 策略3：避免死区移动
            if not any(s['action'] == action for s in strategies):
                strategies.append({
                    'action': action,
                    'strategy_name': '🤔 探索移动',
                    'priority_score': 10,
                    'reasoning': f'移动到({target_pos})，探索新的可能性'
                })
        
        return strategies if strategies else [{'action': valid_actions[0], 'strategy_name': '🎲 随机应变', 'priority_score': 1, 'reasoning': '无明确策略，随机选择'}]
    
    def _is_valid_push_position_llm(self, pos, board, boxes):
        """检查推箱子位置是否有效"""
        row, col = pos
        
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
    
    def _calculate_target_improvement_llm(self, old_pos, new_pos, targets, boxes):
        """计算推动箱子后与目标的距离改善"""
        available_targets = [t for t in targets if t not in boxes]
        if not available_targets:
            return 0
        
        old_min_dist = min(abs(old_pos[0] - t[0]) + abs(old_pos[1] - t[1]) for t in available_targets)
        new_min_dist = min(abs(new_pos[0] - t[0]) + abs(new_pos[1] - t[1]) for t in available_targets)
        
        return max(0, old_min_dist - new_min_dist)
    
    def _calculate_approach_value_llm(self, new_player_pos, incomplete_boxes, targets, urgency_factor):
        """计算接近箱子的价值"""
        if not incomplete_boxes:
            return 0
        
        # 找到最有价值的箱子（最接近目标的箱子）
        box_values = []
        for box in incomplete_boxes:
            available_targets = [t for t in targets if t not in incomplete_boxes]
            if available_targets:
                min_box_to_target = min(abs(box[0] - t[0]) + abs(box[1] - t[1]) for t in available_targets)
                player_to_box = abs(new_player_pos[0] - box[0]) + abs(new_player_pos[1] - box[1])
                
                # 价值 = 目标近度 + 可接近性
                value = max(0, 20 - min_box_to_target) + max(0, 10 - player_to_box)
                box_values.append(value)
        
        return max(box_values) * urgency_factor if box_values else 0
    
    def _gomoku_reasoning(self, observation: Dict[str, Any], env) -> str:
        """五子棋游戏推理"""
        valid_actions = self._get_valid_actions(observation, env)
        
        if not valid_actions:
            return "ACTION: PASS"
        
        # 简单策略：随机选择
        best_action = random.choice(valid_actions)
        
        return f"""
五子棋分析：
- 评估棋局态势
- 寻找最佳落子点
- 选择动作：{best_action}

ACTION: {best_action}
"""
    
    def _default_reasoning(self, observation: Dict[str, Any], env) -> str:
        """默认推理"""
        valid_actions = self._get_valid_actions(observation, env)
        best_action = random.choice(valid_actions) if valid_actions else "PASS"
        
        return f"ACTION: {best_action}"
    
    def call_llm(self, prompt: str) -> str:
        """调用真实的LLM API"""
        # 这里可以实现对OpenAI API或其他LLM的调用
        # 目前返回模拟响应
        if self.api_key:
            try:
                # 实际的API调用逻辑
                # import openai
                # response = openai.ChatCompletion.create(...)
                pass
            except Exception as e:
                print(f"LLM API调用失败: {e}")
        
        # 备用：返回模拟响应
        return "ACTION: UP"  # 简单的备用响应
    
    def parse_action_from_response(self, response: str, valid_actions: List[str]) -> Optional[str]:
        """从LLM响应中解析动作"""
        try:
            # 查找ACTION:标记
            if "ACTION:" in response:
                action_part = response.split("ACTION:")[-1].strip()
                action = action_part.split()[0].strip().upper()
                
                # 检查动作是否有效
                if action in valid_actions:
                    return action
            
            # 尝试从响应中找到有效动作
            response_upper = response.upper()
            for action in valid_actions:
                if action.upper() in response_upper:
                    return action
            
            # 如果都找不到，返回None
            return None
            
        except Exception as e:
            print(f"解析动作失败: {e}")
            return None
    
    def _get_valid_actions(self, observation: Dict[str, Any], env) -> List[str]:
        """获取有效动作"""
        if 'valid_actions_mask' in observation:
            mask = observation['valid_actions_mask']
            actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
            valid_actions = []
            
            try:
                if hasattr(mask, '__len__') and len(mask) >= 4:
                    for i in range(min(4, len(mask))):
                        if bool(mask[i]):
                            valid_actions.append(actions[i])
                else:
                    valid_actions = actions
            except:
                valid_actions = actions
            
            return valid_actions if valid_actions else actions
        else:
            return ['UP', 'DOWN', 'LEFT', 'RIGHT']
    
    def _calculate_min_distance(self, pos1: Optional[Tuple[int, int]], 
                               positions: List[Tuple[int, int]]) -> int:
        """计算到最近位置的距离"""
        if not pos1 or not positions:
            return float('inf')
        
        return min(abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) for pos2 in positions)
    
    def _analyze_gomoku_threats(self, board: np.ndarray, player_id: int) -> List[Tuple[int, int]]:
        """分析五子棋威胁"""
        # 简化的威胁分析
        threats = []
        # 这里可以实现更复杂的威胁检测逻辑
        return threats
    
    def _analyze_gomoku_opportunities(self, board: np.ndarray, player_id: int) -> List[Tuple[int, int]]:
        """分析五子棋机会"""
        # 简化的机会分析
        opportunities = []
        # 这里可以实现更复杂的机会检测逻辑
        return opportunities


class AdvancedSokobanAI(BaseAgent):
    """高级推箱子AI - 混合LLM推理与搜索算法"""
    
    def __init__(self, name: str = "Advanced Sokoban AI", player_id: int = 1, **kwargs):
        super().__init__(name, player_id)
        self.strategy = kwargs.get('strategy', 'hybrid')  # 'llm', 'search', 'hybrid'
        self.reasoning_depth = kwargs.get('reasoning_depth', 3)
        self.search_depth = kwargs.get('search_depth', 15)
        self.state_memory = {}  # 记住状态
        
    def get_action(self, observation: Dict[str, Any], env) -> Optional[str]:
        """获取动作 - 混合策略"""
        try:
            valid_actions = self._get_valid_actions(observation, env)
            if not valid_actions:
                return None
            
            if self.strategy == 'hybrid':
                return self._hybrid_decision(observation, env, valid_actions)
            elif self.strategy == 'llm':
                return self._llm_decision(observation, env, valid_actions)
            elif self.strategy == 'search':
                return self._search_decision(observation, env, valid_actions)
            else:
                return random.choice(valid_actions)
                
        except Exception as e:
            print(f"Advanced Sokoban AI error: {e}")
            valid_actions = self._get_valid_actions(observation, env)
            return random.choice(valid_actions) if valid_actions else None
    
    def observation_to_text(self, observation: Dict[str, Any], env) -> str:
        """将游戏状态转换为文本描述（简化版）"""
        board = observation['board']
        return f"推箱子游戏状态，棋盘大小：{board.shape[0]}x{board.shape[1]}"
    
    def _build_prompt(self, game_description: str, valid_actions: List[str], env) -> str:
        """构建LLM提示词（简化版）"""
        return f"游戏描述：{game_description}\n可选动作：{valid_actions}\n请选择最佳动作。"
    
    def call_llm(self, prompt: str) -> str:
        """调用LLM（简化版）"""
        return "ACTION: RIGHT"  # 简化版本，总是返回RIGHT
    
    def parse_action_from_response(self, response: str, valid_actions: List[str]) -> Optional[str]:
        """从响应中解析动作"""
        if "ACTION:" in response:
            action_part = response.split("ACTION:")[-1].strip()
            action = action_part.split()[0] if action_part.split() else ""
            if action in valid_actions:
                return action
        return None
    
    def _hybrid_decision(self, observation: Dict[str, Any], env, valid_actions: List[str]) -> str:
        """混合决策 - 结合LLM推理和搜索算法"""
        # 获取游戏状态
        state_info = self._analyze_game_state(observation)
        
        # 根据游戏复杂度选择策略
        if state_info['completion_ratio'] > 0.8:
            # 接近完成时，使用精确搜索
            action = self._precision_search(observation, env, valid_actions)
            if action:
                return action
        
        if state_info['immediate_win_available']:
            # 有立即获胜机会，使用搜索确认
            action = self._search_decision(observation, env, valid_actions)
            if action:
                return action
        
        # 默认使用增强LLM推理
        return self._llm_decision(observation, env, valid_actions)
    
    def _analyze_game_state(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """分析游戏状态"""
        board = observation['board']
        
        # 统计箱子和目标
        boxes = []
        targets = []
        
        for row in range(board.shape[0]):
            for col in range(board.shape[1]):
                cell = board[row, col]
                if cell in [3, 4]:  # 箱子
                    boxes.append((row, col))
                if cell in [2, 4, 7, 8]:  # 目标点
                    targets.append((row, col))
        
        completed_boxes = [box for box in boxes if box in targets]
        completion_ratio = len(completed_boxes) / max(1, len(targets))
        
        # 检查是否有立即获胜机会
        immediate_win = len(boxes) - len(completed_boxes) == 1
        
        return {
            'boxes': boxes,
            'targets': targets,
            'completed_boxes': completed_boxes,
            'completion_ratio': completion_ratio,
            'immediate_win_available': immediate_win
        }
    
    def _precision_search(self, observation: Dict[str, Any], env, valid_actions: List[str]) -> Optional[str]:
        """精确搜索 - 用于游戏后期的高精度搜索"""
        player_pos = self._get_player_position(observation)
        if not player_pos:
            return None
        
        state_info = self._analyze_game_state(observation)
        boxes = state_info['boxes']
        targets = state_info['targets']
        board = observation['board']
        
        # 使用多层精确搜索策略
        # 第1层：检查是否有直接获胜的动作
        winning_action = self._find_winning_move(player_pos, boxes, targets, board, valid_actions)
        if winning_action:
            return winning_action
        
        # 第2层：防止死锁的智能推动
        smart_push_action = self._find_smart_push_action(player_pos, boxes, targets, board, valid_actions)
        if smart_push_action:
            return smart_push_action
        
        # 第3层：战略定位搜索
        positioning_action = self._find_positioning_action(player_pos, boxes, targets, board, valid_actions)
        if positioning_action:
            return positioning_action
        
        # 第4层：最有前景的推箱子动作（带死锁检测）
        return self._find_best_push_action(player_pos, boxes, targets, board, valid_actions)
        
        directions = {
            'UP': (-1, 0),
            'DOWN': (1, 0),
            'LEFT': (0, -1),
            'RIGHT': (0, 1)
        }
        
        state_info = self._analyze_game_state(observation)
        boxes = state_info['boxes']
        targets = state_info['targets']
        
        # 寻找可以直接完成的动作
        for action in valid_actions:
            if action in directions:
                dr, dc = directions[action]
                adjacent_pos = (player_pos[0] + dr, player_pos[1] + dc)
                
                if adjacent_pos in boxes:
                    box_new_pos = (adjacent_pos[0] + dr, adjacent_pos[1] + dc)
                    if box_new_pos in targets:
                        return action
        
        # TODO: 添加更多精确搜索逻辑
        
        return None
    
    def _llm_decision(self, observation: Dict[str, Any], env, valid_actions: List[str]) -> str:
        """LLM决策 - 使用增强的LLM推理"""
        game_description = self.observation_to_text(observation, env)
        prompt = self._build_prompt(game_description, valid_actions, env)
        
        # 调用LLM获取决策
        response = self.call_llm(prompt)
        action = self.parse_action_from_response(response, valid_actions)
        
        return action if action else random.choice(valid_actions)
    
    def _search_decision(self, observation: Dict[str, Any], env, valid_actions: List[str]) -> str:
        """搜索决策 - 使用搜索算法寻找最佳动作"""
        # 简化的搜索逻辑：寻找推箱子机会
        player_pos = self._get_player_position(observation)
        if not player_pos:
            return random.choice(valid_actions)
        
        state_info = self._analyze_game_state(observation)
        
        # 寻找可以推箱子的动作
        directions = {
            'UP': (-1, 0),
            'DOWN': (1, 0),
            'LEFT': (0, -1),
            'RIGHT': (0, 1)
        }
        
        for action in valid_actions:
            if action in directions:
                dr, dc = directions[action]
                adjacent_pos = (player_pos[0] + dr, player_pos[1] + dc)
                
                if adjacent_pos in state_info['boxes']:
                    return action
        
        return random.choice(valid_actions)
    
    def _get_valid_actions(self, observation: Dict[str, Any], env) -> List[str]:
        """获取有效动作"""
        if 'valid_actions_mask' in observation:
            mask = observation['valid_actions_mask']
            actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
            valid_actions = []
            
            try:
                if hasattr(mask, '__len__') and len(mask) >= 4:
                    for i in range(min(4, len(mask))):
                        if bool(mask[i]):
                            valid_actions.append(actions[i])
                else:
                    valid_actions = actions
            except:
                valid_actions = actions
            
            return valid_actions if valid_actions else actions
        else:
            return ['UP', 'DOWN', 'LEFT', 'RIGHT']
    
    def _get_player_position(self, observation: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        """获取玩家位置"""
        if self.player_id == 1:
            pos = observation.get('player1_pos', [-1, -1])
        else:
            pos = observation.get('player2_pos', [-1, -1])
        
        if pos[0] >= 0 and pos[1] >= 0:
            return tuple(pos)
        return None
    
    def _find_winning_move(self, player_pos: Tuple[int, int], boxes: List[Tuple[int, int]], 
                          targets: List[Tuple[int, int]], board: np.ndarray, 
                          valid_actions: List[str]) -> Optional[str]:
        """寻找能够直接获胜的移动"""
        directions = {
            'UP': (-1, 0),
            'DOWN': (1, 0),
            'LEFT': (0, -1),
            'RIGHT': (0, 1)
        }
        
        incomplete_boxes = [box for box in boxes if box not in targets]
        
        # 如果只剩一个箱子未完成，寻找直接完成的动作
        if len(incomplete_boxes) == 1:
            box = incomplete_boxes[0]
            available_targets = [t for t in targets if t not in boxes or t == box]
            
            for action in valid_actions:
                if action in directions:
                    dr, dc = directions[action]
                    # 检查是否能推动这个箱子
                    if (player_pos[0] + dr, player_pos[1] + dc) == box:
                        new_box_pos = (box[0] + dr, box[1] + dc)
                        if new_box_pos in available_targets and self._is_valid_position(new_box_pos, board, boxes):
                            return action
        
        return None
    
    def _find_best_push_action(self, player_pos: Tuple[int, int], boxes: List[Tuple[int, int]], 
                              targets: List[Tuple[int, int]], board: np.ndarray, 
                              valid_actions: List[str]) -> Optional[str]:
        """寻找最佳的推箱子动作"""
        directions = {
            'UP': (-1, 0),
            'DOWN': (1, 0),
            'LEFT': (0, -1),
            'RIGHT': (0, 1)
        }
        
        best_action = None
        best_score = -1
        
        for action in valid_actions:
            if action in directions:
                dr, dc = directions[action]
                adjacent_pos = (player_pos[0] + dr, player_pos[1] + dc)
                
                if adjacent_pos in boxes:
                    new_box_pos = (adjacent_pos[0] + dr, adjacent_pos[1] + dc)
                    if self._is_valid_position(new_box_pos, board, boxes):
                        # 计算推动后的价值
                        score = self._evaluate_push_move(adjacent_pos, new_box_pos, targets, boxes)
                        if score > best_score:
                            best_score = score
                            best_action = action
        
        return best_action
    
    def _is_valid_position(self, pos: Tuple[int, int], board: np.ndarray, boxes: List[Tuple[int, int]]) -> bool:
        """检查位置是否有效"""
        row, col = pos
        
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
    
    def _evaluate_push_move(self, old_box_pos: Tuple[int, int], new_box_pos: Tuple[int, int], 
                           targets: List[Tuple[int, int]], boxes: List[Tuple[int, int]]) -> float:
        """评估推动箱子的价值"""
        # 如果推到目标点，最高分
        if new_box_pos in targets:
            return 1000.0
        
        # 计算到最近目标的距离改善
        available_targets = [t for t in targets if t not in boxes or t == old_box_pos]
        if not available_targets:
            return 0.0
        
        old_min_dist = min(abs(old_box_pos[0] - t[0]) + abs(old_box_pos[1] - t[1]) for t in available_targets)
        new_min_dist = min(abs(new_box_pos[0] - t[0]) + abs(new_box_pos[1] - t[1]) for t in available_targets)
        
        improvement = old_min_dist - new_min_dist
        return max(0, improvement * 10)  # 距离改善转换为分数
    
    def _find_smart_push_action(self, player_pos: Tuple[int, int], boxes: List[Tuple[int, int]], 
                               targets: List[Tuple[int, int]], board: np.ndarray, 
                               valid_actions: List[str]) -> Optional[str]:
        """智能推动策略 - 避免死锁的推动"""
        directions = {
            'UP': (-1, 0),
            'DOWN': (1, 0),
            'LEFT': (0, -1),
            'RIGHT': (0, 1)
        }
        
        best_action = None
        best_score = -1
        
        for action in valid_actions:
            if action in directions:
                dr, dc = directions[action]
                adjacent_pos = (player_pos[0] + dr, player_pos[1] + dc)
                
                if adjacent_pos in boxes:
                    new_box_pos = (adjacent_pos[0] + dr, adjacent_pos[1] + dc)
                    
                    # 检查推动是否有效且不会造成死锁
                    if (self._is_valid_position(new_box_pos, board, boxes) and
                        not self._causes_deadlock(new_box_pos, boxes, targets, board)):
                        
                        # 计算推动的战略价值
                        score = self._evaluate_strategic_push(adjacent_pos, new_box_pos, targets, boxes, board)
                        if score > best_score:
                            best_score = score
                            best_action = action
        
        return best_action if best_score > 0 else None
    
    def _find_positioning_action(self, player_pos: Tuple[int, int], boxes: List[Tuple[int, int]], 
                                targets: List[Tuple[int, int]], board: np.ndarray, 
                                valid_actions: List[str]) -> Optional[str]:
        """战略定位 - 移动到最佳推动位置"""
        directions = {
            'UP': (-1, 0),
            'DOWN': (1, 0),
            'LEFT': (0, -1),
            'RIGHT': (0, 1)
        }
        
        # 找到所有可推动的有价值位置
        valuable_positions = []
        incomplete_boxes = [box for box in boxes if box not in targets]
        
        for box in incomplete_boxes:
            # 检查箱子四周的推动位置
            for dr, dc in directions.values():
                push_from_pos = (box[0] - dr, box[1] - dc)  # 玩家需要到达的位置
                push_to_pos = (box[0] + dr, box[1] + dc)    # 箱子将被推到的位置
                
                if (self._is_valid_position(push_from_pos, board, boxes) and
                    self._is_valid_position(push_to_pos, board, boxes) and
                    not self._causes_deadlock(push_to_pos, boxes, targets, board)):
                    
                    # 评估这个推动的价值
                    value = self._evaluate_strategic_push(box, push_to_pos, targets, boxes, board)
                    if value > 0:
                        distance = abs(player_pos[0] - push_from_pos[0]) + abs(player_pos[1] - push_from_pos[1])
                        valuable_positions.append((push_from_pos, value, distance))
        
        if not valuable_positions:
            return None
        
        # 选择价值最高且距离合理的位置
        best_position = max(valuable_positions, key=lambda x: x[1] - x[2] * 0.1)
        target_pos = best_position[0]
        
        # 朝目标位置移动
        best_action = None
        min_distance = float('inf')
        
        for action in valid_actions:
            if action in directions:
                dr, dc = directions[action]
                new_pos = (player_pos[0] + dr, player_pos[1] + dc)
                
                if self._is_valid_position(new_pos, board, boxes):
                    distance = abs(new_pos[0] - target_pos[0]) + abs(new_pos[1] - target_pos[1])
                    if distance < min_distance:
                        min_distance = distance
                        best_action = action
        
        return best_action
    
    def _causes_deadlock(self, box_pos: Tuple[int, int], boxes: List[Tuple[int, int]], 
                        targets: List[Tuple[int, int]], board: np.ndarray) -> bool:
        """检查推动是否会导致死锁"""
        row, col = box_pos
        
        # 基本死锁检测：角落死锁
        if self._is_corner_deadlock(box_pos, board, targets):
            return True
        
        # 墙边死锁检测
        if self._is_wall_deadlock(box_pos, board, targets):
            return True
        
        # 多箱子死锁检测
        if self._is_multi_box_deadlock(box_pos, boxes, board, targets):
            return True
        
        return False
    
    def _is_corner_deadlock(self, pos: Tuple[int, int], board: np.ndarray, targets: List[Tuple[int, int]]) -> bool:
        """检查角落死锁"""
        if pos in targets:
            return False  # 在目标位置不算死锁
        
        row, col = pos
        # 检查四个角落情况
        corners = [
            (row - 1, col - 1), (row - 1, col + 1),  # 上左, 上右
            (row + 1, col - 1), (row + 1, col + 1)   # 下左, 下右
        ]
        
        for corner_row, corner_col in corners:
            # 检查是否被两面墙围住
            wall_top = row - 1 < 0 or board[row - 1, col] == 1
            wall_bottom = row + 1 >= board.shape[0] or board[row + 1, col] == 1
            wall_left = col - 1 < 0 or board[row, col - 1] == 1
            wall_right = col + 1 >= board.shape[1] or board[row, col + 1] == 1
            
            if (wall_top and wall_left) or (wall_top and wall_right) or \
               (wall_bottom and wall_left) or (wall_bottom and wall_right):
                return True
        
        return False
    
    def _is_wall_deadlock(self, pos: Tuple[int, int], board: np.ndarray, targets: List[Tuple[int, int]]) -> bool:
        """检查墙边死锁"""
        if pos in targets:
            return False
        
        row, col = pos
        
        # 检查是否贴着墙且在该方向上没有目标
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dr, dc in directions:
            wall_row, wall_col = row + dr, col + dc
            
            # 如果这个方向是墙
            if (wall_row < 0 or wall_row >= board.shape[0] or 
                wall_col < 0 or wall_col >= board.shape[1] or 
                board[wall_row, wall_col] == 1):
                
                # 检查沿着墙的方向是否有目标
                perpendicular_dirs = [(dc, dr), (-dc, -dr)] if dr == 0 else [(dr, dc), (-dr, -dc)]
                
                has_target_along_wall = False
                for perp_dr, perp_dc in perpendicular_dirs:
                    check_pos = (row + perp_dr, col + perp_dc)
                    if check_pos in targets:
                        has_target_along_wall = True
                        break
                
                if not has_target_along_wall:
                    return True
        
        return False
    
    def _is_multi_box_deadlock(self, pos: Tuple[int, int], boxes: List[Tuple[int, int]], 
                              board: np.ndarray, targets: List[Tuple[int, int]]) -> bool:
        """检查多箱子死锁（简化版）"""
        # 这是一个复杂的问题，这里实现简化版本
        # 检查是否形成了无法移动的箱子群
        
        row, col = pos
        adjacent_boxes = 0
        
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            adj_pos = (row + dr, col + dc)
            if adj_pos in boxes:
                adjacent_boxes += 1
        
        # 如果有太多相邻的箱子，可能形成死锁
        return adjacent_boxes >= 2 and pos not in targets
    
    def _evaluate_strategic_push(self, old_pos: Tuple[int, int], new_pos: Tuple[int, int], 
                                targets: List[Tuple[int, int]], boxes: List[Tuple[int, int]], 
                                board: np.ndarray) -> float:
        """评估战略推动的价值"""
        # 如果推到目标点，最高分
        if new_pos in targets:
            return 1000.0
        
        # 计算到最近目标的距离改善
        available_targets = [t for t in targets if t not in boxes or t == old_pos]
        if not available_targets:
            return 0.0
        
        old_min_dist = min(abs(old_pos[0] - t[0]) + abs(old_pos[1] - t[1]) for t in available_targets)
        new_min_dist = min(abs(new_pos[0] - t[0]) + abs(new_pos[1] - t[1]) for t in available_targets)
        
        distance_improvement = old_min_dist - new_min_dist
        
        # 考虑推动后的位置灵活性
        flexibility_bonus = self._evaluate_position_flexibility(new_pos, board, boxes)
        
        # 考虑是否为其他箱子让出空间
        space_bonus = self._evaluate_space_creation(old_pos, new_pos, boxes, targets)
        
        total_score = distance_improvement * 10 + flexibility_bonus + space_bonus
        return max(0, total_score)
    
    def _evaluate_position_flexibility(self, pos: Tuple[int, int], board: np.ndarray, 
                                     boxes: List[Tuple[int, int]]) -> float:
        """评估位置的灵活性（可移动方向数）"""
        row, col = pos
        flexible_directions = 0
        
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            adj_pos = (row + dr, col + dc)
            if self._is_valid_position(adj_pos, board, boxes):
                flexible_directions += 1
        
        return flexible_directions * 5  # 每个可移动方向加5分
    
    def _evaluate_space_creation(self, old_pos: Tuple[int, int], new_pos: Tuple[int, int], 
                                boxes: List[Tuple[int, int]], targets: List[Tuple[int, int]]) -> float:
        """评估为其他箱子创造空间的价值"""
        # 检查移动后是否为其他箱子到目标创造了路径
        space_value = 0
        
        for box in boxes:
            if box != old_pos:
                for target in targets:
                    if target not in boxes:
                        # 简化检查：如果旧位置阻挡了路径，移动后得分
                        if self._is_on_path(old_pos, box, target) and not self._is_on_path(new_pos, box, target):
                            space_value += 20
        
        return space_value
    
    def _is_on_path(self, pos: Tuple[int, int], start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """简单检查位置是否在起点到终点的直线路径上"""
        # 检查是否在水平或垂直直线路径上
        if start[0] == end[0] and pos[0] == start[0]:  # 同一行
            return min(start[1], end[1]) < pos[1] < max(start[1], end[1])
        elif start[1] == end[1] and pos[1] == start[1]:  # 同一列
            return min(start[0], end[0]) < pos[0] < max(start[0], end[0])
        return False
