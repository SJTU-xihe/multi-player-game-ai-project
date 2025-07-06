"""
进阶AI实现示例（选做）
包含强化学习和大语言模型AI的模板实现
仅供有兴趣的同学参考和扩展
"""

import random
import pickle
import sys
import os
from typing import List, Tuple, Any, Optional

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseAgent


class QLearningBot(BaseAgent):
    """
    Q-Learning强化学习AI
    适合有机器学习基础的同学尝试
    """
    
    def __init__(self, name: str = "QLearningBot", player_id: int = 1):
        super().__init__(name, player_id)
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1  # 探索率
        self.training = True
        
    def get_action(self, observation: Any, env: Any) -> Any:
        """Q-learning策略选择"""
        state = self.observation_to_state(observation)
        valid_actions = env.get_valid_actions()
        
        if not valid_actions:
            return None
        
        # ε-贪心策略
        if self.training and random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # 选择Q值最高的动作
        q_values = {}
        for action in valid_actions:
            q_values[action] = self.q_table.get((state, action), 0.0)
        
        if not q_values:
            return random.choice(valid_actions)
        
        best_action = max(q_values, key=q_values.get)
        return best_action
    
    def observation_to_state(self, observation: Any) -> Tuple:
        """
        将观察转换为状态表示
        这是Q-learning的关键部分，需要设计合适的状态表示
        """
        # 简化状态表示（学生可以改进）
        try:
            if isinstance(observation, dict) and 'board' in observation:
                # 五子棋：使用棋盘的简化表示
                board = observation['board']
                # 为了减少状态空间，只考虑关键区域
                center = board.shape[0] // 2
                key_region = board[max(0, center-3):center+4, max(0, center-3):center+4]
                return tuple(key_region.flatten())
            elif isinstance(observation, dict):
                # 贪吃蛇：使用关键信息
                snake = observation.get(f'snake{self.player_id}', [])
                foods = observation.get('foods', [])
                # 简化表示：只考虑头部位置和最近的食物
                head = snake[0] if snake else (0, 0)
                nearest_food = foods[0] if foods else (0, 0)
                return (head, nearest_food)
            else:
                # 简化处理：转换为简单状态
                return (0, 0)
        except Exception:
            # 异常情况下返回默认状态
            return (0, 0)
    
    def update_q_value(self, state: Tuple, action: Any, reward: float, next_state: Tuple):
        """更新Q值"""
        current_q = self.q_table.get((state, action), 0.0)
        
        # 获取下一状态的最大Q值
        max_next_q = 0.0
        # 这里简化处理，实际应该获取next_state的所有可能动作
        for next_action in self.get_sample_actions():
            q_val = self.q_table.get((next_state, next_action), 0.0)
            max_next_q = max(max_next_q, q_val)
        
        # Q-learning更新公式
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[(state, action)] = new_q
    
    def get_sample_actions(self) -> List[Any]:
        """获取示例动作（简化实现）"""
        # 这里返回一些典型动作，实际应该根据游戏动态获取
        return [(i, j) for i in range(3) for j in range(3)]
    
    def train_episode(self, env: Any, opponent: Any):
        """训练一个回合"""
        observation, _ = env.reset()
        state = self.observation_to_state(observation)
        total_reward = 0
        
        while not env.is_terminal():
            # 获取动作
            if env.game.current_player == self.player_id:
                action = self.get_action(observation, env)
                current_state = state
            else:
                action = opponent.get_action(observation, env)
                current_state = None
            
            # 执行动作
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = self.observation_to_state(next_obs)
            done = terminated or truncated
            
            # 更新Q值（只有当前玩家）
            if current_state is not None:
                self.update_q_value(current_state, action, reward, next_state)
                total_reward += reward
            
            # 更新状态
            state = next_state
            observation = next_obs
            
            if done:
                break
        
        return total_reward
    
    def train(self, env: Any, opponent: Any, episodes: int = 1000):
        """训练Q-learning智能体"""
        print(f"开始训练 {episodes} 轮...")
        self.training = True
        
        for episode in range(episodes):
            total_reward = self.train_episode(env, opponent)
            
            # 衰减探索率
            if episode % 100 == 0:
                self.epsilon = max(0.01, self.epsilon * 0.995)
                print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.3f}")
        
        self.training = False
        print("训练完成！")
    
    def save_model(self, filename: str):
        """保存训练好的模型"""
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.q_table, f)
            print(f"模型已保存到: {filename}")
        except Exception as e:
            print(f"保存模型失败: {e}")
    
    def load_model(self, filename: str):
        """加载训练好的模型"""
        try:
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            self.training = False
            print(f"成功加载模型: {filename}")
        except FileNotFoundError:
            print(f"模型文件不存在: {filename}")
        except Exception as e:
            print(f"加载模型失败: {e}")


class LLMBot(BaseAgent):
    """
    大语言模型AI Bot
    需要API密钥或本地模型支持
    """
    
    def __init__(self, name: str = "LLMBot", player_id: int = 1, model_type: str = "simulate"):
        super().__init__(name, player_id)
        self.model_type = model_type  # "openai", "ollama", "simulate"
        self.api_key = None  # 设置你的API密钥
        self.max_retries = 3
        
    def get_action(self, observation: Any, env: Any) -> Any:
        """使用大语言模型进行决策"""
        try:
            # 转换游戏状态为文字描述
            game_description = self.observation_to_text(observation, env)
            
            # 构建提示词
            prompt = self.build_prompt(game_description, env)
            
            # 调用大模型
            for attempt in range(self.max_retries):
                try:
                    response = self.call_llm(prompt)
                    action = self.parse_action(response, env)
                    
                    if action and action in env.get_valid_actions():
                        return action
                except Exception as e:
                    print(f"LLM调用尝试 {attempt + 1} 失败: {e}")
                    continue
            
            # 如果所有尝试都失败，使用降级策略
            return self.fallback_strategy(observation, env)
                
        except Exception as e:
            print(f"LLM决策失败: {e}")
            return self.fallback_strategy(observation, env)
    
    def observation_to_text(self, observation: Any, env: Any) -> str:
        """将游戏状态转换为文字描述"""
        if hasattr(env, 'board_size'):  # 五子棋类游戏
            # 处理不同格式的观察值
            if isinstance(observation, dict):
                board = observation['board']
            else:
                board = observation
            description = f"棋盘大小: {board.shape[0]}x{board.shape[1]}\n"
            
            # 描述棋盘状态
            description += "当前棋盘状态:\n"
            symbols = {0: "·", 1: "●", 2: "○"}
            
            for i in range(min(board.shape[0], 15)):  # 限制显示大小
                row_desc = ""
                for j in range(min(board.shape[1], 15)):
                    row_desc += symbols.get(board[i, j], "?")
                description += f"{i:2d}|{row_desc}|\n"
            
            # 添加列标号
            col_numbers = "  " + "".join([str(i%10) for i in range(min(board.shape[1], 15))])
            description += col_numbers + "\n"
            
            return description
            
        else:  # 贪吃蛇类游戏
            snake1 = observation.get('snake1', [])
            snake2 = observation.get('snake2', [])
            foods = observation.get('foods', [])
            
            description = f"贪吃蛇游戏状态:\n"
            my_snake = snake1 if self.player_id == 1 else snake2
            opponent_snake = snake2 if self.player_id == 1 else snake1
            
            description += f"我的蛇(玩家{self.player_id}): {my_snake[:5]}{'...' if len(my_snake) > 5 else ''}\n"
            description += f"对手的蛇: {opponent_snake[:5]}{'...' if len(opponent_snake) > 5 else ''}\n"
            description += f"食物位置: {foods}\n"
            
            return description
    
    def build_prompt(self, game_description: str, env: Any) -> str:
        """构建给大模型的提示词"""
        game_name = env.__class__.__name__.replace('Env', '')
        valid_actions = env.get_valid_actions()
        
        # 限制动作显示数量，避免prompt过长
        if len(valid_actions) > 10:
            action_display = f"{valid_actions[:10]}... (共{len(valid_actions)}个动作)"
        else:
            action_display = str(valid_actions)
        
        prompt = f"""你是一个专业的{game_name}游戏AI。

{game_description}

你是玩家{self.player_id}。请分析当前局势并选择最佳动作。

可选动作: {action_display}

请按以下格式回复:
分析: [简要分析当前局势]
动作: (row, col)

要求:
1. 只能从可选动作中选择
2. 动作格式必须是(row, col)的形式
3. 优先考虑获胜机会
4. 其次考虑阻止对手获胜
5. 选择安全且有战略价值的位置
"""
        return prompt
    
    def call_llm(self, prompt: str) -> str:
        """调用大语言模型"""
        if self.model_type == "openai":
            return self.call_openai(prompt)
        elif self.model_type == "ollama":
            return self.call_ollama(prompt)
        else:
            # 模拟模式：使用简单规则生成回复
            return self.simulate_llm_response(prompt)
    
    def call_openai(self, prompt: str) -> str:
        """调用OpenAI API"""
        import requests
        
        if not self.api_key:
            raise ValueError("需要设置OpenAI API密钥")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 200
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"API调用失败: {response.status_code}")
    
    def call_ollama(self, prompt: str) -> str:
        """调用本地Ollama模型"""
        import requests
        
        data = {
            "model": "llama2",  # 或其他已安装的模型
            "prompt": prompt,
            "stream": False
        }
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            raise Exception(f"Ollama调用失败: {response.status_code}")
    
    def simulate_llm_response(self, prompt: str) -> str:
        """模拟大模型回复（用于测试）"""
        # 简单的规则模拟LLM回复
        responses = [
            "分析: 选择中心位置有利于控制局面\n动作: (4, 4)",
            "分析: 阻止对手形成威胁\n动作: (3, 5)",
            "分析: 扩展自己的势力范围\n动作: (5, 3)",
            "分析: 保持攻守平衡\n动作: (6, 6)"
        ]
        return random.choice(responses)
    
    def parse_action(self, response: str, env: Any) -> Optional[Tuple[int, int]]:
        """从大模型回复中解析动作"""
        import re
        
        # 使用正则表达式提取动作
        patterns = [
            r'动作[:：]\s*\((\d+),\s*(\d+)\)',
            r'动作[:：]\s*\((\d+)\s*,\s*(\d+)\)',
            r'\((\d+),\s*(\d+)\)',
            r'(\d+),\s*(\d+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                try:
                    row, col = int(matches[-1][0]), int(matches[-1][1])
                    return (row, col)
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def fallback_strategy(self, observation: Any, env: Any) -> Any:
        """降级策略：当LLM失败时使用"""
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            return None
        
        # 简单的启发式策略
        if hasattr(env, 'board_size'):
            # 五子棋：优先选择中心位置
            center = env.board_size // 2
            for action in valid_actions:
                row, col = action
                if abs(row - center) <= 2 and abs(col - center) <= 2:
                    return action
        
        # 兜底：随机选择
        return random.choice(valid_actions)


def demo_q_learning():
    """演示Q-learning训练过程"""
    print("=== Q-Learning演示 ===")
    
    try:
        from games.gomoku import GomokuEnv
        from agents import RandomBot
        
        # 创建环境
        env = GomokuEnv(board_size=9, win_length=5)
        
        # 创建智能体
        q_bot = QLearningBot("Q-Learning Bot", 1)
        random_bot = RandomBot("Random Bot", 2)
        
        print("开始训练...")
        # 简短训练（完整训练需要更多轮次）
        q_bot.train(env, random_bot, episodes=100)
        
        # 测试训练效果
        print("测试训练效果...")
        wins = 0
        test_games = 10
        
        for i in range(test_games):
            observation, _ = env.reset()
            
            while not env.is_terminal():
                if env.game.current_player == 1:
                    action = q_bot.get_action(observation, env)
                else:
                    action = random_bot.get_action(observation, env)
                
                observation, reward, terminated, truncated, done = env.step(action)
                done = terminated or truncated
                if done:
                    break
            
            if env.get_winner() == 1:
                wins += 1
        
        print(f"训练后胜率: {wins}/{test_games} = {wins/test_games:.1%}")
        
    except Exception as e:
        print(f"Q-Learning演示失败: {e}")


def demo_llm_bot():
    """演示大语言模型AI"""
    print("=== LLM Bot演示 ===")
    
    try:
        from games.gomoku import GomokuEnv
        
        # 创建环境
        env = GomokuEnv(board_size=9, win_length=5)
        
        # 创建LLM Bot（使用模拟模式）
        llm_bot = LLMBot("LLM Bot", 1, model_type="simulate")
        
        # 如果要使用真实API，取消下面的注释并设置API密钥
        # llm_bot.api_key = "your-openai-api-key"
        # llm_bot.model_type = "openai"
        
        observation, _ = env.reset()
        
        for i in range(3):  # 演示几步
            print(f"\n--- 第 {i+1} 步 ---")
            action = llm_bot.get_action(observation, env)
            print(f"LLM选择的动作: {action}")
            
            if action:
                observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                env.render()
                if done:
                    break
            else:
                break
        
    except Exception as e:
        print(f"LLM Bot演示失败: {e}")


if __name__ == "__main__":
    print("进阶AI示例演示")
    print("=" * 50)
    
    # 演示Q-Learning
    demo_q_learning()
    
    print("\n" + "=" * 50)
    
    # 演示LLM Bot
    demo_llm_bot()
    
    print("\n" + "=" * 50)
    print("演示完成！")
    print("\n提示：")
    print("1. Q-Learning需要大量训练才能看到效果")
    print("2. LLM Bot需要API密钥才能使用真实大模型")
    print("3. 这些只是基础示例，学生可以在此基础上改进") 