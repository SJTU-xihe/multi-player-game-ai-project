"""
推箱子游戏专用GUI
支持双人对战和AI对战
"""

import pygame
import sys
import time
import os
from typing import Optional, Tuple, Dict, Any, List
from games.sokoban import SokobanGame, SokobanEnv
from agents import HumanAgent
from agents.ai_bots.sokoban_ai import SokobanAI, SimpleSokobanAI
from agents.ai_bots.llm_bot import LLMBot, AdvancedSokobanAI

# 颜色定义
COLORS = {
    'WHITE': (255, 255, 255),
    'BLACK': (0, 0, 0),
    'RED': (255, 0, 0),
    'BLUE': (0, 0, 255),
    'GREEN': (0, 255, 0),
    'YELLOW': (255, 255, 0),
    'ORANGE': (255, 165, 0),
    'PURPLE': (128, 0, 128),
    'CYAN': (0, 255, 255),
    'BROWN': (139, 69, 19),
    'GRAY': (128, 128, 128),
    'LIGHT_GRAY': (211, 211, 211),
    'DARK_GRAY': (64, 64, 64),
    'WALL_COLOR': (101, 67, 33),
    'FLOOR_COLOR': (245, 245, 220),
    'TARGET_COLOR': (255, 192, 203),
    'BOX_COLOR': (160, 82, 45),
    'BOX_ON_TARGET_COLOR': (255, 69, 0)
}


class SokobanGUI:
    """推箱子图形界面"""
    
    def __init__(self):
        # 初始化pygame
        pygame.init()
        
        # 游戏设置
        self.cell_size = 40
        self.margin = 50
        self.ui_width = 300
        
        # 默认窗口大小（会根据关卡调整）
        self.window_width = 800
        self.window_height = 600
        
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Sokoban AI Battle")
        self.clock = pygame.time.Clock()
        
        # 字体
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        
        # 游戏状态
        self.current_level = 1
        self.game_mode = 'competitive'  # 'competitive' or 'cooperative'
        self.env = None
        self.human_agent = HumanAgent(name="Human Player", player_id=1)
        self.ai_agent = SokobanAI(name="Search AI", player_id=2, max_search_time=2.0, use_dynamic_depth=True)
        self.current_agent = self.human_agent
        self.game_over = False
        self.winner = None
        self.thinking = False
        self.paused = False
        self.selected_ai = "SearchAI"  # 新增AI选项：SearchAI, SimpleAI, LLMAI, HybridAI
        
        # 先创建UI元素（需要在初始化游戏前创建）
        self.buttons = self._create_buttons()
        
        # 创建AI代理
        self._create_ai_agent()
        
        # 初始化游戏
        self._init_game()
        
        # 游戏计时
        self.last_update = time.time()
        self.update_interval = 0.5  # 500ms更新一次
        
    def _init_game(self):
        """初始化游戏"""
        try:
            self.env = SokobanEnv(level_id=self.current_level, game_mode=self.game_mode)
            self._adjust_window_size()
            self.reset_game()
        except Exception as e:
            print(f"Failed to initialize game: {e}")
            # 使用默认设置
            self.env = SokobanEnv(level_id=1, game_mode='competitive')
            self.reset_game()
    
    def _adjust_window_size(self):
        """根据关卡大小调整窗口尺寸"""
        if self.env and self.env.game:
            game_width = self.env.game.width * self.cell_size + self.margin * 2
            game_height = self.env.game.height * self.cell_size + self.margin * 2
            
            self.window_width = game_width + self.ui_width
            self.window_height = max(game_height, 600)
            
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            
            # 更新按钮位置
            self._update_button_positions()
    
    def _update_button_positions(self):
        """更新按钮位置"""
        if hasattr(self, 'buttons') and self.buttons:
            button_width = 120
            button_height = 30
            start_x = self.window_width - self.ui_width + 20
            
            # 更新所有按钮的位置
            y_positions = {
                'sokoban_ai': 50,
                'simple_ai': 90,
                'competitive': 140,
                'cooperative': 180,
                'level_prev': 230,
                'level_next': 230,
                'new_game': 280,
                'pause': 320,
                'hint': 360,
                'quit': 400
            }
            
            for button_name, button_info in self.buttons.items():
                if button_name in y_positions:
                    y = y_positions[button_name]
                    if button_name == 'level_next':
                        button_info['rect'] = pygame.Rect(start_x + 70, y, 50, button_height)
                    elif button_name == 'level_prev':
                        button_info['rect'] = pygame.Rect(start_x, y, 50, button_height)
                    else:
                        button_info['rect'] = pygame.Rect(start_x, y, button_width, button_height)
    
    def _create_buttons(self) -> Dict[str, Dict[str, Any]]:
        """创建UI按钮"""
        button_width = 120
        button_height = 30
        start_x = 650  # 使用固定值，稍后会更新位置
        
        buttons = {
            # AI选择 - 扩展为四种AI
            'search_ai': {
                'rect': pygame.Rect(start_x, 30, button_width, button_height),
                'text': 'Search AI',
                'color': COLORS['YELLOW']
            },
            'llm_ai': {
                'rect': pygame.Rect(start_x, 65, button_width, button_height),
                'text': 'LLM AI',
                'color': COLORS['LIGHT_GRAY']
            },
            'hybrid_ai': {
                'rect': pygame.Rect(start_x, 100, button_width, button_height),
                'text': 'Hybrid AI',
                'color': COLORS['LIGHT_GRAY']
            },
            'simple_ai': {
                'rect': pygame.Rect(start_x, 135, button_width, button_height),
                'text': 'Simple AI',
                'color': COLORS['LIGHT_GRAY']
            },
            
            # 游戏模式
            'competitive': {
                'rect': pygame.Rect(start_x, 175, button_width, button_height),
                'text': 'Competitive',
                'color': COLORS['YELLOW']
            },
            'cooperative': {
                'rect': pygame.Rect(start_x, 210, button_width, button_height),
                'text': 'Cooperative',
                'color': COLORS['LIGHT_GRAY']
            },
            
            # 关卡选择
            'level_prev': {
                'rect': pygame.Rect(start_x, 250, 50, button_height),
                'text': '<',
                'color': COLORS['CYAN']
            },
            'level_next': {
                'rect': pygame.Rect(start_x + 70, 250, 50, button_height),
                'text': '>',
                'color': COLORS['CYAN']
            },
            
            # 控制按钮
            'new_game': {
                'rect': pygame.Rect(start_x, 290, button_width, button_height),
                'text': 'New Game',
                'color': COLORS['GREEN']
            },
            'pause': {
                'rect': pygame.Rect(start_x, 325, button_width, button_height),
                'text': 'Pause',
                'color': COLORS['ORANGE']
            },
            'hint': {
                'rect': pygame.Rect(start_x, 360, button_width, button_height),
                'text': 'Hint',
                'color': COLORS['PURPLE']
            },
            'quit': {
                'rect': pygame.Rect(start_x, 400, button_width, button_height),
                'text': 'Quit',
                'color': COLORS['RED']
            }
        }
        
        return buttons
    
    def _create_ai_agent(self):
        """创建AI智能体"""
        if self.selected_ai == "SearchAI":
            self.ai_agent = SokobanAI(
                name="Search AI", 
                player_id=2,
                max_search_time=2.0,
                use_dynamic_depth=True,
                cache_size=10000
            )
        elif self.selected_ai == "LLMAI":
            self.ai_agent = LLMBot(
                name="LLM AI", 
                player_id=2,
                use_local_simulation=True,
                reasoning_depth=3
            )
        elif self.selected_ai == "HybridAI":
            self.ai_agent = AdvancedSokobanAI(
                name="Hybrid AI", 
                player_id=2,
                strategy='hybrid',
                search_depth=3
            )
        elif self.selected_ai == "SimpleAI":
            self.ai_agent = SimpleSokobanAI(name="Simple AI", player_id=2)
    
    def reset_game(self):
        """重置游戏"""
        if self.env:
            self.env.reset()
            self.game_over = False
            self.winner = None
            self.thinking = False
            self.current_agent = self.human_agent
            self.last_update = time.time()
            self.paused = False
            self.buttons['pause']['text'] = 'Pause'
    
    def handle_events(self) -> bool:
        """处理事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                # 处理推箱子的键盘输入
                if (isinstance(self.current_agent, HumanAgent) and 
                    not self.game_over and not self.thinking and not self.paused):
                    self._handle_sokoban_input(event.key)
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键点击
                    mouse_pos = pygame.mouse.get_pos()
                    self._handle_button_click(mouse_pos)
        
        return True
    
    def _handle_button_click(self, mouse_pos: Tuple[int, int]):
        """处理按钮点击"""
        for button_name, button_info in self.buttons.items():
            if button_info['rect'].collidepoint(mouse_pos):
                if button_name == 'new_game':
                    self.reset_game()
                elif button_name == 'quit':
                    pygame.quit()
                    sys.exit()
                elif button_name == 'pause':
                    self.paused = not self.paused
                    self.buttons['pause']['text'] = 'Resume' if self.paused else 'Pause'
                elif button_name == 'hint':
                    self._show_hint()
                elif button_name == 'level_prev':
                    self._change_level(-1)
                elif button_name == 'level_next':
                    self._change_level(1)
                elif button_name in ['search_ai', 'llm_ai', 'hybrid_ai', 'simple_ai']:
                    # 更新选中的AI
                    for btn_name in ['search_ai', 'llm_ai', 'hybrid_ai', 'simple_ai']:
                        self.buttons[btn_name]['color'] = COLORS['LIGHT_GRAY']
                    
                    if button_name == 'search_ai':
                        self.selected_ai = "SearchAI"
                    elif button_name == 'llm_ai':
                        self.selected_ai = "LLMAI"
                    elif button_name == 'hybrid_ai':
                        self.selected_ai = "HybridAI"
                    elif button_name == 'simple_ai':
                        self.selected_ai = "SimpleAI"
                    
                    self.buttons[button_name]['color'] = COLORS['YELLOW']
                    self._create_ai_agent()
                    self.reset_game()
                elif button_name in ['competitive', 'cooperative']:
                    # 更新游戏模式
                    for btn_name in ['competitive', 'cooperative']:
                        self.buttons[btn_name]['color'] = COLORS['LIGHT_GRAY']
                    
                    self.game_mode = button_name
                    self.buttons[button_name]['color'] = COLORS['YELLOW']
                    self._init_game()
    
    def _change_level(self, direction: int):
        """改变关卡"""
        try:
            available_levels = self.env.get_available_levels() if self.env else []
            if available_levels:
                max_level = max(level['id'] for level in available_levels)
                min_level = min(level['id'] for level in available_levels)
                
                new_level = self.current_level + direction
                if min_level <= new_level <= max_level:
                    self.current_level = new_level
                    self._init_game()
        except Exception as e:
            print(f"Failed to change level: {e}")
    
    def _show_hint(self):
        """显示提示"""
        if self.env and not self.game_over:
            # 使用AI获取提示
            try:
                observation = self.env._get_observation()
                hint_ai = SokobanAI(name="Hint AI", player_id=self.env.get_current_player())
                hint_action = hint_ai.get_action(observation, self.env)
                if hint_action:
                    print(f"Hint: Try moving {hint_action}")
            except Exception as e:
                print(f"Failed to get hint: {e}")
    
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
    
    def _make_move(self, action):
        """执行移动"""
        if self.game_over or self.paused or not self.env:
            return
        
        try:
            # 执行动作
            observation, reward, terminated, truncated, info = self.env.step(action)
            
            # 检查游戏是否结束
            if terminated or truncated:
                self.game_over = True
                self.winner = self.env.get_winner()
            else:
                # 更新当前代理
                current_player = self.env.get_current_player()
                if current_player == 1:
                    self.current_agent = self.human_agent
                else:
                    self.current_agent = self.ai_agent
        
        except Exception as e:
            print(f"Move execution failed: {e}")
    
    def update_game(self):
        """更新游戏状态"""
        if self.game_over or self.paused or not self.env:
            return
        
        current_time = time.time()
        
        # 检查是否需要更新
        if current_time - self.last_update < self.update_interval:
            return
        
        self.last_update = current_time
        
        # 确保当前代理与游戏中的当前玩家匹配
        current_player = self.env.get_current_player()
        if current_player == 1:
            self.current_agent = self.human_agent
        else:
            self.current_agent = self.ai_agent
        
        # 只有AI回合才自动移动
        if not isinstance(self.current_agent, HumanAgent):
            try:
                self.thinking = True
                observation = self.env._get_observation()
                action = self.current_agent.get_action(observation, self.env)
                
                if action:
                    self._make_move(action)
                
            except Exception as e:
                print(f"AI thinking failed: {e}")
            finally:
                self.thinking = False
    
    def draw(self):
        """绘制游戏界面"""
        # 清空屏幕
        self.screen.fill(COLORS['WHITE'])
        
        # 绘制游戏区域
        self._draw_sokoban_game()
        
        # 绘制UI
        self._draw_ui()
        
        # 绘制游戏状态
        self._draw_game_status()
        
        # 更新显示
        pygame.display.flip()
    
    def _draw_sokoban_game(self):
        """绘制推箱子游戏"""
        if not self.env or not self.env.game:
            return
        
        # 绘制游戏区域背景
        game_width = self.env.game.width * self.cell_size
        game_height = self.env.game.height * self.cell_size
        game_rect = pygame.Rect(self.margin, self.margin, game_width, game_height)
        pygame.draw.rect(self.screen, COLORS['FLOOR_COLOR'], game_rect)
        pygame.draw.rect(self.screen, COLORS['BLACK'], game_rect, 2)
        
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
                    self._draw_cell(rect, cell)
    
    def _draw_cell(self, rect: pygame.Rect, cell: str):
        """绘制单个单元格"""
        # 先绘制地面
        pygame.draw.rect(self.screen, COLORS['FLOOR_COLOR'], rect)
        
        # 根据内容绘制
        if cell == '#':  # 墙壁
            pygame.draw.rect(self.screen, COLORS['WALL_COLOR'], rect)
            pygame.draw.rect(self.screen, COLORS['BLACK'], rect, 1)
        
        elif cell == '.':  # 目标点
            pygame.draw.rect(self.screen, COLORS['TARGET_COLOR'], rect)
            pygame.draw.circle(self.screen, COLORS['RED'], rect.center, 8, 2)
        
        elif cell == '$':  # 箱子
            pygame.draw.rect(self.screen, COLORS['BOX_COLOR'], rect)
            pygame.draw.rect(self.screen, COLORS['BLACK'], rect, 2)
            # 绘制箱子纹理
            pygame.draw.line(self.screen, COLORS['BLACK'], 
                           (rect.left + 5, rect.top + 5), 
                           (rect.right - 5, rect.bottom - 5), 1)
            pygame.draw.line(self.screen, COLORS['BLACK'], 
                           (rect.right - 5, rect.top + 5), 
                           (rect.left + 5, rect.bottom - 5), 1)
        
        elif cell == '*':  # 箱子在目标上
            pygame.draw.rect(self.screen, COLORS['BOX_ON_TARGET_COLOR'], rect)
            pygame.draw.rect(self.screen, COLORS['BLACK'], rect, 2)
            # 绘制完成标记
            pygame.draw.circle(self.screen, COLORS['GREEN'], rect.center, 10)
        
        elif cell == '@':  # 玩家1
            pygame.draw.circle(self.screen, COLORS['BLUE'], rect.center, 15)
            pygame.draw.circle(self.screen, COLORS['WHITE'], rect.center, 12)
            pygame.draw.circle(self.screen, COLORS['BLUE'], rect.center, 8)
            # 绘制数字1
            text = self.font_small.render('1', True, COLORS['WHITE'])
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)
        
        elif cell == '&':  # 玩家2
            pygame.draw.circle(self.screen, COLORS['RED'], rect.center, 15)
            pygame.draw.circle(self.screen, COLORS['WHITE'], rect.center, 12)
            pygame.draw.circle(self.screen, COLORS['RED'], rect.center, 8)
            # 绘制数字2
            text = self.font_small.render('2', True, COLORS['WHITE'])
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)
        
        elif cell == '+':  # 玩家1在目标上
            pygame.draw.rect(self.screen, COLORS['TARGET_COLOR'], rect)
            pygame.draw.circle(self.screen, COLORS['BLUE'], rect.center, 15)
            pygame.draw.circle(self.screen, COLORS['WHITE'], rect.center, 12)
            pygame.draw.circle(self.screen, COLORS['BLUE'], rect.center, 8)
            text = self.font_small.render('1', True, COLORS['WHITE'])
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)
        
        elif cell == '%':  # 玩家2在目标上
            pygame.draw.rect(self.screen, COLORS['TARGET_COLOR'], rect)
            pygame.draw.circle(self.screen, COLORS['RED'], rect.center, 15)
            pygame.draw.circle(self.screen, COLORS['WHITE'], rect.center, 12)
            pygame.draw.circle(self.screen, COLORS['RED'], rect.center, 8)
            text = self.font_small.render('2', True, COLORS['WHITE'])
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)
        
        # 绘制网格线
        pygame.draw.rect(self.screen, COLORS['GRAY'], rect, 1)
    
    def _draw_ui(self):
        """绘制UI界面"""
        start_x = self.window_width - self.ui_width + 20
        
        # 绘制按钮
        for button_name, button_info in self.buttons.items():
            pygame.draw.rect(self.screen, button_info['color'], button_info['rect'])
            pygame.draw.rect(self.screen, COLORS['BLACK'], button_info['rect'], 2)
            
            text_surface = self.font_medium.render(button_info['text'], True, COLORS['BLACK'])
            text_rect = text_surface.get_rect(center=button_info['rect'].center)
            self.screen.blit(text_surface, text_rect)
        
        # 绘制标题
        ai_title_text = self.font_medium.render("AI Selection:", True, COLORS['BLACK'])
        self.screen.blit(ai_title_text, (start_x, 5))
        
        mode_title_text = self.font_medium.render("Game Mode:", True, COLORS['BLACK'])
        self.screen.blit(mode_title_text, (start_x, 150))
        
        level_title_text = self.font_medium.render(f"Level: {self.current_level}", True, COLORS['BLACK'])
        self.screen.blit(level_title_text, (start_x, 225))
        
        # 绘制操作说明
        instructions = [
            "Controls:",
            "• Arrow keys/WASD to move",
            "• Push boxes to targets",
            "• Blue player is you",
            "• Red player is AI",
            "",
            "Symbols:",
            "• Brown squares = walls",
            "• Pink circles = targets", 
            "• Brown boxes = boxes",
            "• Red boxes = completed"
        ]
        
        start_y = 450
        for i, instruction in enumerate(instructions):
            text = self.font_small.render(instruction, True, COLORS['DARK_GRAY'])
            self.screen.blit(text, (start_x, start_y + i * 16))
    
    def _draw_game_status(self):
        """绘制游戏状态"""
        start_x = self.window_width - self.ui_width + 20
        status_y = 160
        
        if not self.env:
            return
        
        if self.paused:
            status_text = "Game Paused"
            color = COLORS['ORANGE']
        elif self.game_over:
            if self.winner == 1:
                status_text = "You Win!"
                color = COLORS['GREEN']
            elif self.winner == 2:
                status_text = "AI Wins!"
                color = COLORS['RED']
            elif self.winner == 0:
                status_text = "Draw!" if self.game_mode == 'competitive' else "Success!"
                color = COLORS['ORANGE'] if self.game_mode == 'competitive' else COLORS['GREEN']
            else:
                status_text = "Game Over"
                color = COLORS['GRAY']
        else:
            current_player = self.env.get_current_player()
            if current_player == 1:
                status_text = "Your Turn"
                color = COLORS['BLUE']
            else:
                if self.thinking:
                    status_text = "AI Thinking..."
                    color = COLORS['ORANGE']
                else:
                    status_text = "AI Turn"
                    color = COLORS['RED']
        
        text_surface = self.font_large.render(status_text, True, color)
        self.screen.blit(text_surface, (start_x, status_y))
        
        # 游戏信息
        if self.env:
            state = self.env.get_game_state()
            info_y = status_y + 40
            
            player1_score_text = f"You: {state['player1_score']} boxes"
            player2_score_text = f"AI: {state['player2_score']} boxes"
            progress_text = f"Progress: {state['completed_boxes']}/{state['total_boxes']}"
            steps_text = f"Steps: P1={state['player1_steps']} P2={state['player2_steps']}"
            
            info_surface = self.font_small.render(player1_score_text, True, COLORS['BLUE'])
            self.screen.blit(info_surface, (start_x, info_y))
            
            info_surface2 = self.font_small.render(player2_score_text, True, COLORS['RED'])
            self.screen.blit(info_surface2, (start_x, info_y + 20))
            
            info_surface3 = self.font_small.render(progress_text, True, COLORS['BLACK'])
            self.screen.blit(info_surface3, (start_x, info_y + 40))
            
            info_surface4 = self.font_small.render(steps_text, True, COLORS['DARK_GRAY'])
            self.screen.blit(info_surface4, (start_x, info_y + 60))
    
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
    print("Starting Sokoban AI Battle...")
    print("Controls:")
    print("- Arrow keys or WASD to move your player")
    print("- Push brown boxes to pink targets")
    print("- Try to complete the level before the AI")
    print("- Choose different AI opponents and game modes")
    print("- Use hints if you get stuck")
    
    try:
        game = SokobanGUI()
        game.run()
    except Exception as e:
        print(f"Game error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
