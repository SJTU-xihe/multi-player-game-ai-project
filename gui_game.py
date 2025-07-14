"""
多游戏图形界面
支持五子棋和贪吃蛇的人机对战，修复中文显示问题
"""

import pygame
import sys
import time
import os
from typing import Optional, Tuple, Dict, Any
from games.gomoku import GomokuGame, GomokuEnv
from games.snake import SnakeGame, SnakeEnv
from agents import RandomBot, MinimaxBot, MCTSBot, HumanAgent, SnakeAI, SmartSnakeAI
from agents.ai_bots.gomoku_minimax_bot import GomokuMinimaxBot
from agents.ai_bots.search_ai import SearchAI
import config

# 尝试导入推箱子游戏
try:
    from games.sokoban import SokobanGame, SokobanEnv
    from agents.ai_bots.sokoban_ai import SokobanAI, SimpleSokobanAI
    SOKOBAN_AVAILABLE = True
except ImportError:
    SOKOBAN_AVAILABLE = False
    print("推箱子游戏模块未找到，将跳过推箱子功能")

# 颜色定义
COLORS = {
    "WHITE": (255, 255, 255),
    "BLACK": (0, 0, 0),
    "BROWN": (139, 69, 19),
    "LIGHT_BROWN": (205, 133, 63),
    "RED": (255, 0, 0),
    "BLUE": (0, 0, 255),
    "GREEN": (0, 255, 0),
    "GRAY": (128, 128, 128),
    "LIGHT_GRAY": (211, 211, 211),
    "DARK_GRAY": (64, 64, 64),
    "YELLOW": (255, 255, 0),
    "ORANGE": (255, 165, 0),
    "PURPLE": (128, 0, 128),
    "CYAN": (0, 255, 255),
}


class MultiGameGUI:
    """多游戏图形界面"""

    def __init__(self):
        # 初始化pygame
        pygame.init()

        # 设置中文字体
        self.font_path = self._get_chinese_font()
        self.font_large = pygame.font.Font(self.font_path, 28)
        self.font_medium = pygame.font.Font(self.font_path, 20)
        self.font_small = pygame.font.Font(self.font_path, 16)

        self.window_width = 900
        self.window_height = 700
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("多游戏AI对战平台")
        self.clock = pygame.time.Clock()

        # 游戏状态
        self.current_game = "gomoku"  # "gomoku", "snake", 或 "sokoban"
        self.current_level = 1  # 当前推箱子关卡
        self.env = None
        self.human_agent = None
        self.ai_agent = None
        self.current_agent = None
        self.game_over = False
        self.winner = None
        self.last_move = None
        self.thinking = False
        self.selected_ai = "RandomBot"
        self.paused = False

        # UI元素
        self.buttons = self._create_buttons()
        self.cell_size = 25
        self.margin = 50

        # 游戏计时
        self.last_update = time.time()
        self.update_interval = 0.3  # 贪吃蛇更新间隔

        self._switch_game("gomoku")

    def _get_chinese_font(self):
        """获取中文字体路径"""
        # 尝试不同系统的中文字体
        font_paths = [
            # macOS
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
            # Windows
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
            "C:/Windows/Fonts/msyh.ttc",
            # Linux
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]

        for font_path in font_paths:
            if os.path.exists(font_path):
                return font_path

        # 如果没有找到中文字体，使用pygame默认字体
        return None

    def _create_buttons(self) -> Dict[str, Dict[str, Any]]:
        """创建UI按钮"""
        button_width = 120
        button_height = 30
        start_x = 650

        buttons = {
            # 游戏选择
            "gomoku_game": {
                "rect": pygame.Rect(start_x, 50, button_width, button_height),
                "text": "Gomoku",
                "color": COLORS["YELLOW"],
            },
            "snake_game": {
                "rect": pygame.Rect(start_x, 90, button_width, button_height),
                "text": "Snake",
                "color": COLORS["LIGHT_GRAY"],
            },
        }
        
        # 添加推箱子游戏按钮（如果可用）
        if SOKOBAN_AVAILABLE:
            buttons["sokoban_game"] = {
                "rect": pygame.Rect(start_x, 130, button_width, button_height),
                "text": "Sokoban",
                "color": COLORS["LIGHT_GRAY"],
            }
            ai_start_y = 170
        else:
            ai_start_y = 150
        
        # AI按钮将根据当前游戏动态创建
        # 初始状态为五子棋，稍后会在_update_ai_buttons中更新
        self.ai_start_y = ai_start_y
        self.button_width = button_width
        self.button_height = button_height
        self.start_x = start_x
        
        # 控制按钮的位置会在_update_ai_buttons中确定
        control_start_y = ai_start_y + 120  # 默认位置
        
        # 控制按钮
        control_buttons = {
            "new_game": {
                "rect": pygame.Rect(start_x, control_start_y, button_width, button_height),
                "text": "New Game",
                "color": COLORS["GREEN"],
            },
            "pause": {
                "rect": pygame.Rect(start_x, control_start_y + 40, button_width, button_height),
                "text": "Pause",
                "color": COLORS["ORANGE"],
            },
            "quit": {
                "rect": pygame.Rect(start_x, control_start_y + 80, button_width, button_height),
                "text": "Quit",
                "color": COLORS["RED"],
            },
        }
        
        # 合并控制按钮到主按钮字典
        buttons.update(control_buttons)

        return buttons

    def _switch_game(self, game_type):
        """切换游戏类型"""
        self.current_game = game_type

        # 更新游戏选择按钮颜色
        game_buttons = ["gomoku_game", "snake_game"]
        if SOKOBAN_AVAILABLE:
            game_buttons.append("sokoban_game")
        
        for btn_name in game_buttons:
            if btn_name in self.buttons:
                self.buttons[btn_name]["color"] = COLORS["LIGHT_GRAY"]
        self.buttons[f"{game_type}_game"]["color"] = COLORS["YELLOW"]

        # 创建对应的环境和智能体
        if game_type == "gomoku":
            self.env = GomokuEnv(board_size=15, win_length=5)
            self.cell_size = 30
            self.update_interval = 1.0  # 五子棋不需要频繁更新
            # 设置五子棋默认AI
            if self.selected_ai not in ["GomokuMinimaxBot", "RandomBot", "SearchBFS", "SearchDFS", "SearchAStar"]:
                self.selected_ai = "GomokuMinimaxBot"
        elif game_type == "snake":
            self.env = SnakeEnv(board_size=20)
            self.cell_size = 25
            self.update_interval = 0.3  # 贪吃蛇需要频繁更新
            # 设置贪吃蛇默认AI
            if self.selected_ai not in ["MinimaxBot", "MCTSBot", "RandomBot", "SearchBFS", "SearchDFS", "SearchAStar"]:
                self.selected_ai = "MinimaxBot"
        elif game_type == "sokoban" and SOKOBAN_AVAILABLE:
            self.env = SokobanEnv(level_id=self.current_level, game_mode='competitive')
            self.cell_size = 40
            self.update_interval = 0.5
            # 设置推箱子默认AI
            if self.selected_ai not in ["SokobanAI", "SimpleSokobanAI", "RandomBot", "SearchBFS", "SearchDFS", "SearchAStar"]:
                self.selected_ai = "SokobanAI"

        # 更新AI按钮显示
        self._update_ai_buttons()
        
        self.human_agent = HumanAgent(name="Human Player", player_id=1)
        self._create_ai_agent()
        self.reset_game()

    def _create_ai_agent(self):
        """创建AI智能体"""
        if self.selected_ai == "RandomBot":
            self.ai_agent = RandomBot(name="Random AI", player_id=2)
        elif self.selected_ai == "MinimaxBot":
            if self.current_game == "gomoku":
                self.ai_agent = MinimaxBot(name="Minimax AI", player_id=2, max_depth=3)
            else:
                self.ai_agent = MinimaxBot(name="Minimax AI", player_id=2, max_depth=3)
        elif self.selected_ai == "MCTSBot":
            if self.current_game == "gomoku":
                self.ai_agent = MCTSBot(
                    name="MCTS AI", player_id=2, simulation_count=300
                )
            else:
                self.ai_agent = MCTSBot(name="MCTS AI", player_id=2, simulation_count=300)
        elif self.selected_ai == "GomokuMinimaxBot":
            if self.current_game == "gomoku":
                self.ai_agent = GomokuMinimaxBot(
                    name="Gomoku Expert AI", player_id=2, max_depth=4, timeout=3.0
                )
            else:
                # 对于贪吃蛇游戏，降级到通用Minimax
                self.ai_agent = MinimaxBot(name="Minimax AI", player_id=2, max_depth=3)
        elif self.selected_ai == "SokobanAI" and SOKOBAN_AVAILABLE:
            if self.current_game == "sokoban":
                self.ai_agent = SokobanAI(name="Smart Sokoban AI", player_id=2)
            else:
                # 降级到随机AI
                self.ai_agent = RandomBot(name="Random AI", player_id=2)
        elif self.selected_ai == "SimpleSokobanAI" and SOKOBAN_AVAILABLE:
            if self.current_game == "sokoban":
                self.ai_agent = SimpleSokobanAI(name="Simple Sokoban AI", player_id=2)
            else:
                # 降级到随机AI
                self.ai_agent = RandomBot(name="Random AI", player_id=2)
        elif self.selected_ai == "SearchBFS":
            # BFS搜索AI
            depth = 15 if self.current_game == "snake" else 10
            self.ai_agent = SearchAI(name="BFS Search AI", player_id=2, 
                                   search_algorithm="bfs", max_depth=depth)
        elif self.selected_ai == "SearchDFS":
            # DFS搜索AI
            depth = 20 if self.current_game == "snake" else 15
            self.ai_agent = SearchAI(name="DFS Search AI", player_id=2, 
                                   search_algorithm="dfs", max_depth=depth)
        elif self.selected_ai == "SearchAStar":
            # A*搜索AI
            depth = 18 if self.current_game == "snake" else 12
            self.ai_agent = SearchAI(name="A* Search AI", player_id=2, 
                                   search_algorithm="astar", max_depth=depth)

    def reset_game(self):
        """重置游戏"""
        self.env.reset()
        self.game_over = False
        self.winner = None
        self.last_move = None
        self.thinking = False
        self.current_agent = self.human_agent
        self.last_update = time.time()
        self.paused = False
        self.buttons["pause"]["text"] = "Pause"

    def handle_events(self) -> bool:
        """处理事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.KEYDOWN:
                # 处理贪吃蛇的键盘输入
                if (
                    self.current_game == "snake"
                    and isinstance(self.current_agent, HumanAgent)
                    and not self.game_over
                    and not self.thinking
                    and not self.paused
                ):
                    self._handle_snake_input(event.key)
                
                # 处理推箱子的键盘输入
                elif (
                    self.current_game == "sokoban"
                    and isinstance(self.current_agent, HumanAgent)
                    and not self.game_over
                    and not self.thinking
                    and not self.paused
                ):
                    self._handle_sokoban_input(event.key)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键点击
                    mouse_pos = pygame.mouse.get_pos()

                    # 检查按钮点击
                    click_result = self._handle_button_click(mouse_pos)
                    if click_result is None:
                        return False
                    elif click_result is True:
                        # 如果点击了按钮，重置游戏状态,避免多余处理
                        self.reset_game()
                    # 检查五子棋棋盘点击
                    if (
                        self.current_game == "gomoku"
                        and not self.game_over
                        and isinstance(self.current_agent, HumanAgent)
                        and not self.thinking
                        and not self.paused
                    ):
                        self._handle_gomoku_click(mouse_pos)

        return True

    def _handle_button_click(self, mouse_pos: Tuple[int, int]) -> bool:
        """处理按钮点击"""
        for button_name, button_info in self.buttons.items():
            if button_info["rect"].collidepoint(mouse_pos):
                if button_name == "new_game":
                    self.reset_game()
                elif button_name == "quit":
                    return None
                elif button_name == "pause":
                    self.paused = not self.paused
                    self.buttons["pause"]["text"] = "Resume" if self.paused else "Pause"
                elif button_name in ["gomoku_game", "snake_game", "sokoban_game"]:
                    game_type = button_name.split("_")[0]
                    self._switch_game(game_type)
                elif button_name.endswith("_ai") or button_name == "simple_sokoban_ai":
                    # 重置所有AI按钮颜色
                    ai_button_names = ["random_ai", "minimax_ai", "mcts_ai", "gomoku_ai", "sokoban_ai", "simple_sokoban_ai",
                                      "search_bfs_ai", "search_dfs_ai", "search_astar_ai"]
                    for ai_btn in ai_button_names:
                        if ai_btn in self.buttons:
                            self.buttons[ai_btn]["color"] = COLORS["LIGHT_GRAY"]

                    # 设置新的AI
                    if button_name == "random_ai":
                        self.selected_ai = "RandomBot"
                    elif button_name == "minimax_ai":
                        self.selected_ai = "MinimaxBot"
                    elif button_name == "mcts_ai":
                        self.selected_ai = "MCTSBot"
                    elif button_name == "gomoku_ai":
                        self.selected_ai = "GomokuMinimaxBot"
                    elif button_name == "sokoban_ai":
                        self.selected_ai = "SokobanAI"
                    elif button_name == "simple_sokoban_ai":
                        self.selected_ai = "SimpleSokobanAI"
                    elif button_name == "search_bfs_ai":
                        self.selected_ai = "SearchBFS"
                    elif button_name == "search_dfs_ai":
                        self.selected_ai = "SearchDFS"
                    elif button_name == "search_astar_ai":
                        self.selected_ai = "SearchAStar"

                    # 高亮选中的按钮
                    self.buttons[button_name]["color"] = COLORS["YELLOW"]
                    self._create_ai_agent()
                    self.reset_game()
                elif button_name == "level_prev":
                    self._change_level(-1)
                elif button_name == "level_next":
                    self._change_level(1)

                return True
        return False

    def _handle_gomoku_click(self, mouse_pos: Tuple[int, int]):
        """处理五子棋棋盘点击"""
        x, y = mouse_pos
        board_x = x - self.margin
        board_y = y - self.margin

        if board_x < 0 or board_y < 0:
            return

        col = round(board_x / self.cell_size)
        row = round(board_y / self.cell_size)

        if 0 <= row < 15 and 0 <= col < 15:
            action = (row, col)
            if action in self.env.get_valid_actions():
                self._make_move(action)

    def _handle_snake_input(self, key):
        """处理贪吃蛇键盘输入"""
        key_to_action = {
            pygame.K_UP: (-1, 0),  # 上
            pygame.K_w: (-1, 0),
            pygame.K_DOWN: (1, 0),  # 下
            pygame.K_s: (1, 0),
            pygame.K_LEFT: (0, -1),  # 左
            pygame.K_a: (0, -1),
            pygame.K_RIGHT: (0, 1),  # 右
            pygame.K_d: (0, 1),
        }

        if key in key_to_action:
            action = key_to_action[key]
            self._make_move(action)

    def _make_move(self, action):
        """执行移动"""
        if self.game_over or self.paused:
            return

        try:
            # 执行动作
            observation, reward, terminated, truncated, info = self.env.step(action)
            self.last_move = action

            # 检查游戏是否结束
            if terminated or truncated:
                self.game_over = True
                self.winner = self.env.get_winner()
            else:
                # 切换玩家
                self._switch_player()

        except Exception as e:
            print(f"Move execution failed: {e}")

    def _handle_sokoban_input(self, key):
        """处理推箱子键盘输入"""
        if not isinstance(self.current_agent, HumanAgent):
            return
            
        key_to_action = {
            pygame.K_UP: 'UP',
            pygame.K_w: 'UP',
            pygame.K_DOWN: 'DOWN',
            pygame.K_s: 'DOWN',
            pygame.K_LEFT: 'LEFT',
            pygame.K_a: 'LEFT',
            pygame.K_RIGHT: 'RIGHT',
            pygame.K_d: 'RIGHT'
        }
        
        if key in key_to_action:
            action = key_to_action[key]
            self._make_move(action)

    def _switch_player(self):
        """切换玩家"""
        if isinstance(self.current_agent, HumanAgent):
            self.current_agent = self.ai_agent
            self.thinking = True
        else:
            self.current_agent = self.human_agent

    def _change_level(self, direction: int):
        """改变推箱子关卡"""
        if self.current_game != "sokoban" or not SOKOBAN_AVAILABLE:
            return
            
        try:
            # 获取可用关卡信息
            if self.env:
                available_levels = self.env.get_available_levels()
                if available_levels:
                    max_level = max(level['id'] for level in available_levels)
                    min_level = min(level['id'] for level in available_levels)
                    
                    new_level = self.current_level + direction
                    if min_level <= new_level <= max_level:
                        self.current_level = new_level
                        # 重新创建Sokoban环境以加载新关卡
                        self.env = SokobanEnv(level_id=self.current_level, game_mode='competitive')
                        self.reset_game()  # 重置游戏状态
                        print(f"切换到关卡 {self.current_level}")
                    else:
                        print(f"关卡范围: {min_level}-{max_level}, 当前: {self.current_level}")
        except Exception as e:
            print(f"关卡切换失败: {e}")

    def update_game(self):
        """更新游戏"""
        if self.game_over or self.paused:
            return

        current_time = time.time()

        # 检查是否需要更新
        if current_time - self.last_update < self.update_interval:
            return

        self.last_update = current_time

        # AI回合
        if not isinstance(self.current_agent, HumanAgent) and self.thinking:
            try:
                # 获取AI动作
                observation = self.env._get_observation()
                action = self.current_agent.get_action(observation, self.env)

                if action:
                    self._make_move(action)

                self.thinking = False

            except Exception as e:
                print(f"AI thinking failed: {e}")
                self.thinking = False

        # 贪吃蛇AI自动更新
        elif (
            self.current_game == "snake"
            and not isinstance(self.current_agent, HumanAgent)  # 只有AI才自动移动
            and not self.thinking
        ):
            # AI玩家自动移动
            try:
                observation = self.env._get_observation()
                action = self.current_agent.get_action(observation, self.env)
                
                if action:
                    self._make_move(action)
                    
            except Exception as e:
                print(f"Snake AI move failed: {e}")
        
        # 人类玩家需要通过键盘输入来控制蛇的移动，不在这里自动移动

    def draw(self):
        """绘制游戏界面"""
        # 清空屏幕
        self.screen.fill(COLORS["WHITE"])

        # 绘制游戏区域
        if self.current_game == "gomoku":
            self._draw_gomoku()
        elif self.current_game == "snake":
            self._draw_snake()
        elif self.current_game == "sokoban" and SOKOBAN_AVAILABLE:
            self._draw_sokoban()

        # 绘制UI
        self._draw_ui()

        # 绘制游戏状态
        self._draw_game_status()

        # 更新显示
        pygame.display.flip()

    def _draw_gomoku(self):
        """绘制五子棋"""
        board_size = 15

        # 绘制棋盘背景
        board_rect = pygame.Rect(
            self.margin - 20,
            self.margin - 20,
            board_size * self.cell_size + 40,
            board_size * self.cell_size + 40,
        )
        pygame.draw.rect(self.screen, COLORS["LIGHT_BROWN"], board_rect)

        # 绘制网格线
        for i in range(board_size):
            # 垂直线
            start_pos = (self.margin + i * self.cell_size, self.margin)
            end_pos = (
                self.margin + i * self.cell_size,
                self.margin + (board_size - 1) * self.cell_size,
            )
            pygame.draw.line(self.screen, COLORS["BLACK"], start_pos, end_pos, 2)

            # 水平线
            start_pos = (self.margin, self.margin + i * self.cell_size)
            end_pos = (
                self.margin + (board_size - 1) * self.cell_size,
                self.margin + i * self.cell_size,
            )
            pygame.draw.line(self.screen, COLORS["BLACK"], start_pos, end_pos, 2)

        # 绘制星位
        star_positions = [(3, 3), (3, 11), (11, 3), (11, 11), (7, 7)]
        for row, col in star_positions:
            center = (
                self.margin + col * self.cell_size,
                self.margin + row * self.cell_size,
            )
            pygame.draw.circle(self.screen, COLORS["BLACK"], center, 4)

        # 绘制棋子
        board = self.env.game.board
        for row in range(board_size):
            for col in range(board_size):
                if board[row, col] != 0:
                    center = (
                        self.margin + col * self.cell_size,
                        self.margin + row * self.cell_size,
                    )

                    if board[row, col] == 1:  # 人类玩家
                        color = COLORS["BLACK"]
                        border_color = COLORS["WHITE"]
                    else:  # AI玩家
                        color = COLORS["WHITE"]
                        border_color = COLORS["BLACK"]

                    pygame.draw.circle(self.screen, color, center, 12)
                    pygame.draw.circle(self.screen, border_color, center, 12, 2)

        # 绘制最后一步标记
        if (
            self.last_move
            and isinstance(self.last_move, tuple)
            and len(self.last_move) == 2
        ):
            row, col = self.last_move
            center = (
                self.margin + col * self.cell_size,
                self.margin + row * self.cell_size,
            )
            pygame.draw.circle(self.screen, COLORS["RED"], center, 6, 3)

    def _draw_snake(self):
        """绘制贪吃蛇"""
        board_size = 20

        # 绘制游戏区域背景
        game_rect = pygame.Rect(
            self.margin,
            self.margin,
            board_size * self.cell_size,
            board_size * self.cell_size,
        )
        pygame.draw.rect(self.screen, COLORS["LIGHT_GRAY"], game_rect)
        pygame.draw.rect(self.screen, COLORS["BLACK"], game_rect, 2)

        # 绘制网格
        for i in range(board_size + 1):
            # 垂直线
            x = self.margin + i * self.cell_size
            pygame.draw.line(
                self.screen,
                COLORS["GRAY"],
                (x, self.margin),
                (x, self.margin + board_size * self.cell_size),
                1,
            )
            # 水平线
            y = self.margin + i * self.cell_size
            pygame.draw.line(
                self.screen,
                COLORS["GRAY"],
                (self.margin, y),
                (self.margin + board_size * self.cell_size, y),
                1,
            )

        # 绘制游戏元素
        game = self.env.game
        
        # 绘制蛇1
        if hasattr(game, 'snake1') and game.snake1:
            for i, (row, col) in enumerate(game.snake1):
                if 0 <= row < board_size and 0 <= col < board_size:
                    x = self.margin + col * self.cell_size + 2
                    y = self.margin + row * self.cell_size + 2
                    rect = pygame.Rect(x, y, self.cell_size - 4, self.cell_size - 4)
                    
                    if i == 0:  # 蛇1头部
                        pygame.draw.rect(self.screen, COLORS["BLUE"], rect)
                    else:  # 蛇1身体
                        pygame.draw.rect(self.screen, COLORS["CYAN"], rect)
        
        # 绘制蛇2
        if hasattr(game, 'snake2') and game.snake2:
            for i, (row, col) in enumerate(game.snake2):
                if 0 <= row < board_size and 0 <= col < board_size:
                    x = self.margin + col * self.cell_size + 2
                    y = self.margin + row * self.cell_size + 2
                    rect = pygame.Rect(x, y, self.cell_size - 4, self.cell_size - 4)
                    
                    if i == 0:  # 蛇2头部
                        pygame.draw.rect(self.screen, COLORS["RED"], rect)
                    else:  # 蛇2身体
                        pygame.draw.rect(self.screen, COLORS["ORANGE"], rect)
        
        # 绘制食物
        if hasattr(game, 'foods'):
            for row, col in game.foods:
                if 0 <= row < board_size and 0 <= col < board_size:
                    x = self.margin + col * self.cell_size + 2
                    y = self.margin + row * self.cell_size + 2
                    rect = pygame.Rect(x, y, self.cell_size - 4, self.cell_size - 4)
                    pygame.draw.rect(self.screen, COLORS["GREEN"], rect)

    def _draw_sokoban(self):
        """绘制推箱子游戏"""
        if not self.env or not self.env.game:
            return
        
        # 绘制游戏区域背景
        game_width = self.env.game.width * self.cell_size
        game_height = self.env.game.height * self.cell_size
        game_rect = pygame.Rect(self.margin, self.margin, game_width, game_height)
        pygame.draw.rect(self.screen, (245, 245, 220), game_rect)  # FLOOR_COLOR
        pygame.draw.rect(self.screen, COLORS["BLACK"], game_rect, 2)
        
        # 获取游戏状态
        state = self.env.get_game_state()
        board = state['board']
        
        for row in range(self.env.game.height):
            for col in range(self.env.game.width):
                x = self.margin + col * self.cell_size
                y = self.margin + row * self.cell_size
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                
                # 获取单元格内容
                if row < len(board) and col < len(board[row]):
                    cell = board[row][col]
                    self._draw_sokoban_cell(rect, cell)
    
    def _draw_sokoban_cell(self, rect: pygame.Rect, cell: str):
        """绘制推箱子单个单元格"""
        # 先绘制地面
        pygame.draw.rect(self.screen, (245, 245, 220), rect)  # FLOOR_COLOR
        
        # 根据内容绘制
        if cell == '#':  # 墙壁
            pygame.draw.rect(self.screen, (101, 67, 33), rect)  # WALL_COLOR
            pygame.draw.rect(self.screen, COLORS["BLACK"], rect, 1)
        
        elif cell == '.':  # 目标点
            pygame.draw.rect(self.screen, (255, 192, 203), rect)  # TARGET_COLOR
            pygame.draw.circle(self.screen, COLORS["RED"], rect.center, 8, 2)
        
        elif cell == '$':  # 箱子
            pygame.draw.rect(self.screen, (160, 82, 45), rect)  # BOX_COLOR
            pygame.draw.rect(self.screen, COLORS["BLACK"], rect, 2)
            # 绘制箱子纹理
            pygame.draw.line(self.screen, COLORS["BLACK"], 
                           (rect.left + 5, rect.top + 5), 
                           (rect.right - 5, rect.bottom - 5), 1)
            pygame.draw.line(self.screen, COLORS["BLACK"], 
                           (rect.right - 5, rect.top + 5), 
                           (rect.left + 5, rect.bottom - 5), 1)
        
        elif cell == '*':  # 箱子在目标上
            pygame.draw.rect(self.screen, (255, 69, 0), rect)  # BOX_ON_TARGET_COLOR
            pygame.draw.rect(self.screen, COLORS["BLACK"], rect, 2)
            # 绘制完成标记
            pygame.draw.circle(self.screen, COLORS["GREEN"], rect.center, 10)
        
        elif cell == '@':  # 玩家1
            pygame.draw.circle(self.screen, COLORS["BLUE"], rect.center, 15)
            pygame.draw.circle(self.screen, COLORS["WHITE"], rect.center, 12)
            pygame.draw.circle(self.screen, COLORS["BLUE"], rect.center, 8)
            # 绘制数字1
            text = self.font_small.render('1', True, COLORS["WHITE"])
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)
        
        elif cell == '&':  # 玩家2
            pygame.draw.circle(self.screen, COLORS["RED"], rect.center, 15)
            pygame.draw.circle(self.screen, COLORS["WHITE"], rect.center, 12)
            pygame.draw.circle(self.screen, COLORS["RED"], rect.center, 8)
            # 绘制数字2
            text = self.font_small.render('2', True, COLORS["WHITE"])
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)
        
        elif cell == '+':  # 玩家1在目标上
            pygame.draw.rect(self.screen, (255, 192, 203), rect)  # TARGET_COLOR
            pygame.draw.circle(self.screen, COLORS["BLUE"], rect.center, 15)
            pygame.draw.circle(self.screen, COLORS["WHITE"], rect.center, 12)
            pygame.draw.circle(self.screen, COLORS["BLUE"], rect.center, 8)
            text = self.font_small.render('1', True, COLORS["WHITE"])
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)
        
        elif cell == '%':  # 玩家2在目标上
            pygame.draw.rect(self.screen, (255, 192, 203), rect)  # TARGET_COLOR
            pygame.draw.circle(self.screen, COLORS["RED"], rect.center, 15)
            pygame.draw.circle(self.screen, COLORS["WHITE"], rect.center, 12)
            pygame.draw.circle(self.screen, COLORS["RED"], rect.center, 8)
            text = self.font_small.render('2', True, COLORS["WHITE"])
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)
        
        # 绘制网格线
        pygame.draw.rect(self.screen, COLORS["GRAY"], rect, 1)

    def _draw_ui(self):
        """绘制UI界面"""
        # 绘制按钮
        for button_name, button_info in self.buttons.items():
            pygame.draw.rect(self.screen, button_info["color"], button_info["rect"])
            pygame.draw.rect(self.screen, COLORS["BLACK"], button_info["rect"], 2)

            text_surface = self.font_medium.render(
                button_info["text"], True, COLORS["BLACK"]
            )
            text_rect = text_surface.get_rect(center=button_info["rect"].center)
            self.screen.blit(text_surface, text_rect)

        # 绘制标题
        title_text = self.font_medium.render("Game Selection:", True, COLORS["BLACK"])
        self.screen.blit(title_text, (self.buttons["gomoku_game"]["rect"].x, 25))

        ai_title_text = self.font_medium.render("AI Selection:", True, COLORS["BLACK"])
        self.screen.blit(ai_title_text, (self.buttons["random_ai"]["rect"].x, 125))

        # 绘制操作说明
        if self.current_game == "gomoku":
            instructions = [
                "Gomoku Controls:",
                "• Click to place stone",
                "• Connect 5 to win",
            ]
        elif self.current_game == "snake":
            instructions = [
                "Snake Controls:",
                "• Arrow keys/WASD to move",
                "• Eat food to grow",
                "• Avoid collision",
            ]
        elif self.current_game == "sokoban":
            instructions = [
                "Sokoban Controls:",
                "• Arrow keys/WASD to move",
                "• Push boxes to targets",
                "• Complete all targets",
            ]
        else:
            instructions = [
                "Game Controls:",
                "• Select game type above",
                "• Choose AI opponent",
            ]

        start_y = 450
        for i, instruction in enumerate(instructions):
            text = self.font_small.render(instruction, True, COLORS["DARK_GRAY"])
            self.screen.blit(
                text, (self.buttons["new_game"]["rect"].x, start_y + i * 20)
            )

    def _draw_game_status(self):
        """绘制游戏状态"""
        status_x = 20
        status_y = self.window_height - 100

        if self.paused:
            status_text = "Game Paused..."
            color = COLORS["ORANGE"]
        elif self.game_over:
            if self.winner == 1:
                status_text = "Congratulations! You Win!"
                color = COLORS["GREEN"]
            elif self.winner == 2:
                status_text = "AI Wins! Try Again!"
                color = COLORS["RED"]
            else:
                status_text = "Draw!"
                color = COLORS["ORANGE"]
        else:
            if isinstance(self.current_agent, HumanAgent):
                if self.current_game == "gomoku":
                    status_text = "Your Turn - Click to Place Stone"
                else:
                    status_text = "Your Turn - Use Arrow Keys"
                color = COLORS["BLUE"]
            else:
                if self.thinking:
                    status_text = f"{self.ai_agent.name} is Thinking..."
                    color = COLORS["ORANGE"]
                else:
                    status_text = f"{self.ai_agent.name}'s Turn"
                    color = COLORS["RED"]

        text_surface = self.font_large.render(status_text, True, color)
        self.screen.blit(text_surface, (status_x, status_y))

        # 游戏信息
        info_y = status_y + 40
        if self.current_game == "gomoku":
            player_info = f"Black: Human Player  White: {self.ai_agent.name if self.ai_agent else 'AI'}"
        else:
            if hasattr(self.env.game, "snake1") and hasattr(self.env.game, "snake2"):
                len1 = len(self.env.game.snake1) if self.env.game.alive1 else 0
                len2 = len(self.env.game.snake2) if self.env.game.alive2 else 0
                alive1 = "Alive" if self.env.game.alive1 else "Dead"
                alive2 = "Alive" if self.env.game.alive2 else "Dead"
                player_info = f"Blue Snake(You): {len1} segments({alive1})  Red Snake(AI): {len2} segments({alive2})"
            else:
                player_info = "Snake Battle in Progress..."

        info_surface = self.font_small.render(player_info, True, COLORS["DARK_GRAY"])
        self.screen.blit(info_surface, (status_x, info_y))

    def _update_ai_buttons(self):
        """根据当前游戏类型更新AI按钮"""
        # 移除所有现有的AI按钮
        ai_button_names = ["random_ai", "minimax_ai", "mcts_ai", "gomoku_ai", "sokoban_ai", "simple_sokoban_ai",
                          "search_bfs_ai", "search_dfs_ai", "search_astar_ai"]
        for btn_name in ai_button_names:
            if btn_name in self.buttons:
                del self.buttons[btn_name]
        
        # 根据当前游戏添加相应的AI按钮
        ai_buttons = {}
        y_offset = 0
        
        if self.current_game == "gomoku":
            # 五子棋专用AI按钮
            ai_buttons["gomoku_ai"] = {
                "rect": pygame.Rect(self.start_x, self.ai_start_y + y_offset, self.button_width, self.button_height),
                "text": "Gomoku AI",
                "color": COLORS["YELLOW"] if self.selected_ai == "GomokuMinimaxBot" else COLORS["LIGHT_GRAY"],
            }
            y_offset += 40
            
            # 搜索AI按钮
            ai_buttons["search_bfs_ai"] = {
                "rect": pygame.Rect(self.start_x, self.ai_start_y + y_offset, self.button_width, self.button_height),
                "text": "Search BFS",
                "color": COLORS["YELLOW"] if self.selected_ai == "SearchBFS" else COLORS["LIGHT_GRAY"],
            }
            y_offset += 40
            
            ai_buttons["search_astar_ai"] = {
                "rect": pygame.Rect(self.start_x, self.ai_start_y + y_offset, self.button_width, self.button_height),
                "text": "Search A*",
                "color": COLORS["YELLOW"] if self.selected_ai == "SearchAStar" else COLORS["LIGHT_GRAY"],
            }
            y_offset += 40
            
        elif self.current_game == "snake":
            # 贪吃蛇专用AI按钮
            ai_buttons["minimax_ai"] = {
                "rect": pygame.Rect(self.start_x, self.ai_start_y + y_offset, self.button_width, self.button_height),
                "text": "Minimax AI",
                "color": COLORS["YELLOW"] if self.selected_ai == "MinimaxBot" else COLORS["LIGHT_GRAY"],
            }
            y_offset += 40
            
            ai_buttons["mcts_ai"] = {
                "rect": pygame.Rect(self.start_x, self.ai_start_y + y_offset, self.button_width, self.button_height),
                "text": "MCTS AI",
                "color": COLORS["YELLOW"] if self.selected_ai == "MCTSBot" else COLORS["LIGHT_GRAY"],
            }
            y_offset += 40
            
            # 搜索AI按钮
            ai_buttons["search_bfs_ai"] = {
                "rect": pygame.Rect(self.start_x, self.ai_start_y + y_offset, self.button_width, self.button_height),
                "text": "Search BFS",
                "color": COLORS["YELLOW"] if self.selected_ai == "SearchBFS" else COLORS["LIGHT_GRAY"],
            }
            y_offset += 40
            
            ai_buttons["search_dfs_ai"] = {
                "rect": pygame.Rect(self.start_x, self.ai_start_y + y_offset, self.button_width, self.button_height),
                "text": "Search DFS",
                "color": COLORS["YELLOW"] if self.selected_ai == "SearchDFS" else COLORS["LIGHT_GRAY"],
            }
            y_offset += 40
            
            ai_buttons["search_astar_ai"] = {
                "rect": pygame.Rect(self.start_x, self.ai_start_y + y_offset, self.button_width, self.button_height),
                "text": "Search A*",
                "color": COLORS["YELLOW"] if self.selected_ai == "SearchAStar" else COLORS["LIGHT_GRAY"],
            }
            y_offset += 40
            
        elif self.current_game == "sokoban" and SOKOBAN_AVAILABLE:
            # 推箱子专用AI按钮
            ai_buttons["sokoban_ai"] = {
                "rect": pygame.Rect(self.start_x, self.ai_start_y + y_offset, self.button_width, self.button_height),
                "text": "Smart AI",
                "color": COLORS["YELLOW"] if self.selected_ai == "SokobanAI" else COLORS["LIGHT_GRAY"],
            }
            y_offset += 40
            
            ai_buttons["simple_sokoban_ai"] = {
                "rect": pygame.Rect(self.start_x, self.ai_start_y + y_offset, self.button_width, self.button_height),
                "text": "Simple AI",
                "color": COLORS["YELLOW"] if self.selected_ai == "SimpleSokobanAI" else COLORS["LIGHT_GRAY"],
            }
            y_offset += 40
            
            # 搜索AI按钮
            ai_buttons["search_bfs_ai"] = {
                "rect": pygame.Rect(self.start_x, self.ai_start_y + y_offset, self.button_width, self.button_height),
                "text": "Search BFS",
                "color": COLORS["YELLOW"] if self.selected_ai == "SearchBFS" else COLORS["LIGHT_GRAY"],
            }
            y_offset += 40
            
            # 关卡切换按钮
            ai_buttons["level_prev"] = {
                "rect": pygame.Rect(self.start_x, self.ai_start_y + y_offset, self.button_width // 2 - 5, self.button_height),
                "text": "Prev",
                "color": COLORS["CYAN"],
            }
            ai_buttons["level_next"] = {
                "rect": pygame.Rect(self.start_x + self.button_width // 2 + 5, self.ai_start_y + y_offset, self.button_width // 2 - 5, self.button_height),
                "text": "Next",
                "color": COLORS["CYAN"],
            }
            y_offset += 40
        
        # 通用AI按钮（所有游戏都有）
        ai_buttons["random_ai"] = {
            "rect": pygame.Rect(self.start_x, self.ai_start_y + y_offset, self.button_width, self.button_height),
            "text": "Random AI",
            "color": COLORS["YELLOW"] if self.selected_ai == "RandomBot" else COLORS["LIGHT_GRAY"],
        }
        y_offset += 40
        
        # 更新按钮字典
        self.buttons.update(ai_buttons)
        
        # 更新控制按钮位置
        control_start_y = self.ai_start_y + y_offset + 20
        control_buttons = ["new_game", "pause", "quit"]
        for i, btn_name in enumerate(control_buttons):
            if btn_name in self.buttons:
                self.buttons[btn_name]["rect"] = pygame.Rect(
                    self.start_x, 
                    control_start_y + i * 40, 
                    self.button_width, 
                    self.button_height
                )

    def run(self):
        """运行游戏主循环"""
        running = True

        while running:
            # 处理事件
            running = self.handle_events()

            # 更新游戏
            self.update_game()

            # 绘制界面
            self.draw()

            # 控制帧率
            self.clock.tick(60)

        pygame.quit()
        sys.exit()


def main():
    """主函数"""
    print("Starting Multi-Game AI Battle Platform...")
    print("Supported Games:")
    print("- Gomoku: Click to place stones")
    print("- Snake: Arrow keys/WASD to control")
    print("- Multiple AI difficulty levels")
    print("- Real-time human vs AI battles")

    try:
        game = MultiGameGUI()
        game.run()
    except Exception as e:
        print(f"Game error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
