"""
推箱子关卡编辑器
允许用户创建和编辑自定义关卡
"""

import pygame
import sys
import json
import os
from typing import Dict, List, Tuple, Any, Optional
from games.sokoban.sokoban_game import SokobanGame

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
    'SELECTED_COLOR': (255, 255, 0)
}


class SokobanLevelEditor:
    """推箱子关卡编辑器"""
    
    def __init__(self):
        # 初始化pygame
        pygame.init()
        
        # 编辑器设置
        self.cell_size = 40
        self.margin = 50
        self.ui_width = 250
        self.default_width = 10
        self.default_height = 8
        
        # 窗口设置
        self.window_width = self.default_width * self.cell_size + self.margin * 2 + self.ui_width
        self.window_height = self.default_height * self.cell_size + self.margin * 2 + 100
        
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Sokoban Level Editor")
        self.clock = pygame.time.Clock()
        
        # 字体
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        
        # 编辑状态
        self.current_tool = SokobanGame.WALL  # 当前选择的工具
        self.board_width = self.default_width
        self.board_height = self.default_height
        self.board = [[SokobanGame.FLOOR for _ in range(self.board_width)] 
                     for _ in range(self.board_height)]
        
        # 关卡信息
        self.level_name = "Custom Level"
        self.level_description = "A custom level"
        self.level_difficulty = "medium"
        
        # UI状态
        self.input_mode = None  # 'name', 'description', 'width', 'height'
        self.input_text = ""
        
        # 工具和按钮
        self.tools = self._create_tools()
        self.buttons = self._create_buttons()
        
        # 加载现有关卡数据
        self.levels_data = self._load_levels()
        
    def _load_levels(self) -> Dict:
        """加载现有关卡数据"""
        levels_file = os.path.join('games', 'sokoban', 'levels.json')
        try:
            with open(levels_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"levels": [], "symbols": {}, "rules": {}}
    
    def _save_levels(self):
        """保存关卡数据"""
        levels_file = os.path.join('games', 'sokoban', 'levels.json')
        os.makedirs(os.path.dirname(levels_file), exist_ok=True)
        
        with open(levels_file, 'w', encoding='utf-8') as f:
            json.dump(self.levels_data, f, ensure_ascii=False, indent=2)
    
    def _create_tools(self) -> Dict[str, Dict[str, Any]]:
        """创建编辑工具"""
        tools = {
            SokobanGame.FLOOR: {'name': 'Floor', 'color': COLORS['FLOOR_COLOR']},
            SokobanGame.WALL: {'name': 'Wall', 'color': COLORS['WALL_COLOR']},
            SokobanGame.TARGET: {'name': 'Target', 'color': COLORS['TARGET_COLOR']},
            SokobanGame.BOX: {'name': 'Box', 'color': COLORS['BOX_COLOR']},
            SokobanGame.PLAYER1: {'name': 'Player1', 'color': COLORS['BLUE']},
            SokobanGame.PLAYER2: {'name': 'Player2', 'color': COLORS['RED']},
            SokobanGame.BOX_ON_TARGET: {'name': 'Box+Target', 'color': COLORS['ORANGE']},
            'DELETE': {'name': 'Delete', 'color': COLORS['RED']}
        }
        return tools
    
    def _create_buttons(self) -> Dict[str, Dict[str, Any]]:
        """创建UI按钮"""
        button_width = 100
        button_height = 25
        start_x = self.board_width * self.cell_size + self.margin * 2 + 10
        
        buttons = {
            'new_level': {
                'rect': pygame.Rect(start_x, 50, button_width, button_height),
                'text': 'New Level',
                'color': COLORS['GREEN']
            },
            'save_level': {
                'rect': pygame.Rect(start_x, 80, button_width, button_height),
                'text': 'Save Level',
                'color': COLORS['BLUE']
            },
            'load_level': {
                'rect': pygame.Rect(start_x, 110, button_width, button_height),
                'text': 'Load Level',
                'color': COLORS['CYAN']
            },
            'test_level': {
                'rect': pygame.Rect(start_x, 140, button_width, button_height),
                'text': 'Test Level',
                'color': COLORS['YELLOW']
            },
            'export_level': {
                'rect': pygame.Rect(start_x, 170, button_width, button_height),
                'text': 'Export',
                'color': COLORS['PURPLE']
            },
            'resize_board': {
                'rect': pygame.Rect(start_x, 200, button_width, button_height),
                'text': 'Resize',
                'color': COLORS['ORANGE']
            },
            'validate': {
                'rect': pygame.Rect(start_x, 230, button_width, button_height),
                'text': 'Validate',
                'color': COLORS['LIGHT_GRAY']
            },
            'quit': {
                'rect': pygame.Rect(start_x, 260, button_width, button_height),
                'text': 'Quit',
                'color': COLORS['RED']
            }
        }
        
        return buttons
    
    def handle_events(self) -> bool:
        """处理事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                if self.input_mode:
                    self._handle_text_input(event)
                else:
                    self._handle_keyboard_shortcuts(event)
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键
                    mouse_pos = pygame.mouse.get_pos()
                    self._handle_mouse_click(mouse_pos)
                elif event.button == 3:  # 右键 - 删除
                    mouse_pos = pygame.mouse.get_pos()
                    self._handle_right_click(mouse_pos)
            
            elif event.type == pygame.MOUSEMOTION:
                if pygame.mouse.get_pressed()[0]:  # 左键拖拽
                    mouse_pos = pygame.mouse.get_pos()
                    self._handle_mouse_drag(mouse_pos)
        
        return True
    
    def _handle_text_input(self, event):
        """处理文本输入"""
        if event.key == pygame.K_RETURN:
            self._finish_text_input()
        elif event.key == pygame.K_ESCAPE:
            self.input_mode = None
            self.input_text = ""
        elif event.key == pygame.K_BACKSPACE:
            self.input_text = self.input_text[:-1]
        else:
            if event.unicode.isprintable():
                self.input_text += event.unicode
    
    def _finish_text_input(self):
        """完成文本输入"""
        if self.input_mode == 'name':
            self.level_name = self.input_text or "Custom Level"
        elif self.input_mode == 'description':
            self.level_description = self.input_text or "A custom level"
        elif self.input_mode == 'width':
            try:
                new_width = int(self.input_text)
                if 5 <= new_width <= 30:
                    self._resize_board(new_width, self.board_height)
            except ValueError:
                pass
        elif self.input_mode == 'height':
            try:
                new_height = int(self.input_text)
                if 5 <= new_height <= 20:
                    self._resize_board(self.board_width, new_height)
            except ValueError:
                pass
        
        self.input_mode = None
        self.input_text = ""
    
    def _handle_keyboard_shortcuts(self, event):
        """处理键盘快捷键"""
        if event.key == pygame.K_s and pygame.key.get_pressed()[pygame.K_LCTRL]:
            self._save_current_level()
        elif event.key == pygame.K_n and pygame.key.get_pressed()[pygame.K_LCTRL]:
            self._new_level()
        elif event.key == pygame.K_t and pygame.key.get_pressed()[pygame.K_LCTRL]:
            self._test_level()
        
        # 工具快捷键
        tool_keys = {
            pygame.K_1: SokobanGame.FLOOR,
            pygame.K_2: SokobanGame.WALL,
            pygame.K_3: SokobanGame.TARGET,
            pygame.K_4: SokobanGame.BOX,
            pygame.K_5: SokobanGame.PLAYER1,
            pygame.K_6: SokobanGame.PLAYER2,
            pygame.K_7: SokobanGame.BOX_ON_TARGET,
            pygame.K_DELETE: 'DELETE'
        }
        
        if event.key in tool_keys:
            self.current_tool = tool_keys[event.key]
    
    def _handle_mouse_click(self, mouse_pos: Tuple[int, int]):
        """处理鼠标点击"""
        # 检查工具选择
        tool_y = 350
        tool_height = 30
        start_x = self.board_width * self.cell_size + self.margin * 2 + 10
        
        for i, (tool, info) in enumerate(self.tools.items()):
            tool_rect = pygame.Rect(start_x, tool_y + i * (tool_height + 5), 100, tool_height)
            if tool_rect.collidepoint(mouse_pos):
                self.current_tool = tool
                return
        
        # 检查按钮点击
        for button_name, button_info in self.buttons.items():
            if button_info['rect'].collidepoint(mouse_pos):
                self._handle_button_click(button_name)
                return
        
        # 检查棋盘点击
        self._handle_board_click(mouse_pos)
    
    def _handle_right_click(self, mouse_pos: Tuple[int, int]):
        """处理右键点击（删除）"""
        old_tool = self.current_tool
        self.current_tool = 'DELETE'
        self._handle_board_click(mouse_pos)
        self.current_tool = old_tool
    
    def _handle_mouse_drag(self, mouse_pos: Tuple[int, int]):
        """处理鼠标拖拽"""
        self._handle_board_click(mouse_pos)
    
    def _handle_board_click(self, mouse_pos: Tuple[int, int]):
        """处理棋盘点击"""
        # 计算网格位置
        board_x = mouse_pos[0] - self.margin
        board_y = mouse_pos[1] - self.margin
        
        if board_x < 0 or board_y < 0:
            return
        
        col = board_x // self.cell_size
        row = board_y // self.cell_size
        
        if 0 <= row < self.board_height and 0 <= col < self.board_width:
            if self.current_tool == 'DELETE':
                self.board[row][col] = SokobanGame.FLOOR
            else:
                # 特殊处理：如果放置玩家，先清除其他玩家
                if self.current_tool in [SokobanGame.PLAYER1, SokobanGame.PLAYER2]:
                    self._clear_players(self.current_tool)
                
                self.board[row][col] = self.current_tool
    
    def _clear_players(self, player_type: str):
        """清除指定类型的玩家"""
        for row in range(self.board_height):
            for col in range(self.board_width):
                if self.board[row][col] == player_type:
                    self.board[row][col] = SokobanGame.FLOOR
    
    def _handle_button_click(self, button_name: str):
        """处理按钮点击"""
        if button_name == 'new_level':
            self._new_level()
        elif button_name == 'save_level':
            self._save_current_level()
        elif button_name == 'load_level':
            self._load_level()
        elif button_name == 'test_level':
            self._test_level()
        elif button_name == 'export_level':
            self._export_level()
        elif button_name == 'resize_board':
            self._start_resize()
        elif button_name == 'validate':
            self._validate_level()
        elif button_name == 'quit':
            pygame.quit()
            sys.exit()
    
    def _new_level(self):
        """创建新关卡"""
        self.board = [[SokobanGame.FLOOR for _ in range(self.board_width)] 
                     for _ in range(self.board_height)]
        self.level_name = "Custom Level"
        self.level_description = "A custom level"
    
    def _save_current_level(self):
        """保存当前关卡"""
        # 验证关卡
        validation_result = self._validate_level_data()
        if not validation_result['valid']:
            print(f"无法保存关卡: {validation_result['error']}")
            return
        
        # 转换棋盘为字符串格式
        level_map = []
        for row in self.board:
            level_map.append(''.join(row))
        
        # 创建关卡数据
        new_level = {
            'id': len(self.levels_data.get('levels', [])) + 1,
            'name': self.level_name,
            'difficulty': self.level_difficulty,
            'description': self.level_description,
            'map': level_map
        }
        
        # 添加到关卡列表
        if 'levels' not in self.levels_data:
            self.levels_data['levels'] = []
        
        self.levels_data['levels'].append(new_level)
        
        # 保存到文件
        try:
            self._save_levels()
            print(f"关卡已保存: {self.level_name}")
        except Exception as e:
            print(f"保存失败: {e}")
    
    def _load_level(self):
        """加载关卡（简化版本，加载第一个关卡）"""
        if self.levels_data.get('levels'):
            level = self.levels_data['levels'][0]
            self._load_level_from_data(level)
    
    def _load_level_from_data(self, level_data: Dict):
        """从数据加载关卡"""
        level_map = level_data.get('map', [])
        if not level_map:
            return
        
        # 更新棋盘大小
        self.board_height = len(level_map)
        self.board_width = max(len(row) for row in level_map) if level_map else 10
        
        # 调整窗口大小
        self._resize_window()
        
        # 加载棋盘数据
        self.board = []
        for row_str in level_map:
            row = []
            for i in range(self.board_width):
                if i < len(row_str):
                    row.append(row_str[i])
                else:
                    row.append(SokobanGame.FLOOR)
            self.board.append(row)
        
        # 加载关卡信息
        self.level_name = level_data.get('name', 'Loaded Level')
        self.level_description = level_data.get('description', 'A loaded level')
        self.level_difficulty = level_data.get('difficulty', 'medium')
    
    def _test_level(self):
        """测试关卡"""
        validation_result = self._validate_level_data()
        if not validation_result['valid']:
            print(f"无法测试关卡: {validation_result['error']}")
            return
        
        # 这里可以启动游戏测试关卡
        print("关卡验证通过，可以开始测试")
    
    def _export_level(self):
        """导出关卡到文件"""
        level_map = []
        for row in self.board:
            level_map.append(''.join(row))
        
        export_data = {
            'name': self.level_name,
            'description': self.level_description,
            'difficulty': self.level_difficulty,
            'map': level_map
        }
        
        filename = f"{self.level_name.replace(' ', '_')}.json"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            print(f"关卡已导出到: {filename}")
        except Exception as e:
            print(f"导出失败: {e}")
    
    def _start_resize(self):
        """开始调整棋盘大小"""
        self.input_mode = 'width'
        self.input_text = str(self.board_width)
    
    def _resize_board(self, new_width: int, new_height: int):
        """调整棋盘大小"""
        # 创建新棋盘
        new_board = [[SokobanGame.FLOOR for _ in range(new_width)] 
                    for _ in range(new_height)]
        
        # 复制现有数据
        for row in range(min(self.board_height, new_height)):
            for col in range(min(self.board_width, new_width)):
                new_board[row][col] = self.board[row][col]
        
        self.board = new_board
        self.board_width = new_width
        self.board_height = new_height
        
        # 调整窗口大小
        self._resize_window()
        
        # 重新创建按钮
        self.buttons = self._create_buttons()
    
    def _resize_window(self):
        """调整窗口大小"""
        self.window_width = self.board_width * self.cell_size + self.margin * 2 + self.ui_width
        self.window_height = max(self.board_height * self.cell_size + self.margin * 2 + 100, 600)
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
    
    def _validate_level(self):
        """验证关卡"""
        result = self._validate_level_data()
        print(f"关卡验证: {'通过' if result['valid'] else '失败'}")
        if not result['valid']:
            print(f"错误: {result['error']}")
        else:
            print("关卡设计合理，可以正常游戏")
    
    def _validate_level_data(self) -> Dict[str, Any]:
        """验证关卡数据"""
        # 检查是否有玩家
        player_count = 0
        box_count = 0
        target_count = 0
        
        for row in self.board:
            for cell in row:
                if cell in [SokobanGame.PLAYER1, SokobanGame.PLAYER2]:
                    player_count += 1
                elif cell == SokobanGame.BOX:
                    box_count += 1
                elif cell == SokobanGame.TARGET:
                    target_count += 1
                elif cell == SokobanGame.BOX_ON_TARGET:
                    box_count += 1
                    target_count += 1
        
        # 验证条件
        if player_count == 0:
            return {'valid': False, 'error': '至少需要一个玩家'}
        
        if box_count == 0:
            return {'valid': False, 'error': '至少需要一个箱子'}
        
        if target_count == 0:
            return {'valid': False, 'error': '至少需要一个目标点'}
        
        if box_count != target_count:
            return {'valid': False, 'error': f'箱子数量({box_count})必须等于目标点数量({target_count})'}
        
        return {'valid': True, 'error': None}
    
    def draw(self):
        """绘制编辑器界面"""
        # 清空屏幕
        self.screen.fill(COLORS['WHITE'])
        
        # 绘制棋盘
        self._draw_board()
        
        # 绘制工具栏
        self._draw_tools()
        
        # 绘制按钮
        self._draw_buttons()
        
        # 绘制信息
        self._draw_info()
        
        # 绘制输入提示
        if self.input_mode:
            self._draw_input_prompt()
        
        # 更新显示
        pygame.display.flip()
    
    def _draw_board(self):
        """绘制棋盘"""
        # 绘制背景
        board_rect = pygame.Rect(
            self.margin, 
            self.margin,
            self.board_width * self.cell_size,
            self.board_height * self.cell_size
        )
        pygame.draw.rect(self.screen, COLORS['FLOOR_COLOR'], board_rect)
        pygame.draw.rect(self.screen, COLORS['BLACK'], board_rect, 2)
        
        # 绘制网格和内容
        for row in range(self.board_height):
            for col in range(self.board_width):
                x = self.margin + col * self.cell_size
                y = self.margin + row * self.cell_size
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                
                # 绘制单元格内容
                cell = self.board[row][col]
                self._draw_cell(rect, cell)
                
                # 绘制网格线
                pygame.draw.rect(self.screen, COLORS['GRAY'], rect, 1)
    
    def _draw_cell(self, rect: pygame.Rect, cell: str):
        """绘制单个单元格"""
        if cell == SokobanGame.WALL:
            pygame.draw.rect(self.screen, COLORS['WALL_COLOR'], rect)
        elif cell == SokobanGame.TARGET:
            pygame.draw.rect(self.screen, COLORS['TARGET_COLOR'], rect)
            pygame.draw.circle(self.screen, COLORS['RED'], rect.center, 8, 2)
        elif cell == SokobanGame.BOX:
            pygame.draw.rect(self.screen, COLORS['BOX_COLOR'], rect)
            pygame.draw.rect(self.screen, COLORS['BLACK'], rect, 2)
        elif cell == SokobanGame.PLAYER1:
            pygame.draw.circle(self.screen, COLORS['BLUE'], rect.center, 15)
            text = self.font_small.render('1', True, COLORS['WHITE'])
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)
        elif cell == SokobanGame.PLAYER2:
            pygame.draw.circle(self.screen, COLORS['RED'], rect.center, 15)
            text = self.font_small.render('2', True, COLORS['WHITE'])
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)
        elif cell == SokobanGame.BOX_ON_TARGET:
            pygame.draw.rect(self.screen, COLORS['ORANGE'], rect)
            pygame.draw.rect(self.screen, COLORS['BLACK'], rect, 2)
    
    def _draw_tools(self):
        """绘制工具栏"""
        start_x = self.board_width * self.cell_size + self.margin * 2 + 10
        tool_y = 350
        tool_height = 30
        
        # 标题
        title_text = self.font_medium.render("Tools:", True, COLORS['BLACK'])
        self.screen.blit(title_text, (start_x, tool_y - 25))
        
        # 绘制工具
        for i, (tool, info) in enumerate(self.tools.items()):
            y = tool_y + i * (tool_height + 5)
            tool_rect = pygame.Rect(start_x, y, 100, tool_height)
            
            # 高亮当前选中的工具
            color = COLORS['SELECTED_COLOR'] if tool == self.current_tool else info['color']
            pygame.draw.rect(self.screen, color, tool_rect)
            pygame.draw.rect(self.screen, COLORS['BLACK'], tool_rect, 2)
            
            # 绘制工具名称
            text = self.font_small.render(info['name'], True, COLORS['BLACK'])
            text_rect = text.get_rect(center=tool_rect.center)
            self.screen.blit(text, text_rect)
    
    def _draw_buttons(self):
        """绘制按钮"""
        for button_name, button_info in self.buttons.items():
            pygame.draw.rect(self.screen, button_info['color'], button_info['rect'])
            pygame.draw.rect(self.screen, COLORS['BLACK'], button_info['rect'], 2)
            
            text_surface = self.font_small.render(button_info['text'], True, COLORS['BLACK'])
            text_rect = text_surface.get_rect(center=button_info['rect'].center)
            self.screen.blit(text_surface, text_rect)
    
    def _draw_info(self):
        """绘制关卡信息"""
        start_x = self.board_width * self.cell_size + self.margin * 2 + 10
        info_y = 300
        
        # 关卡信息
        info_lines = [
            f"Name: {self.level_name}",
            f"Size: {self.board_width}x{self.board_height}",
            f"Difficulty: {self.level_difficulty}"
        ]
        
        for i, line in enumerate(info_lines):
            text = self.font_small.render(line, True, COLORS['BLACK'])
            self.screen.blit(text, (start_x, info_y + i * 16))
    
    def _draw_input_prompt(self):
        """绘制输入提示"""
        prompt_texts = {
            'name': f"Enter level name: {self.input_text}",
            'description': f"Enter description: {self.input_text}",
            'width': f"Enter width (5-30): {self.input_text}",
            'height': f"Enter height (5-20): {self.input_text}"
        }
        
        if self.input_mode in prompt_texts:
            prompt_text = prompt_texts[self.input_mode]
            text_surface = self.font_medium.render(prompt_text, True, COLORS['BLACK'])
            
            # 绘制背景
            bg_rect = pygame.Rect(50, 50, text_surface.get_width() + 20, 40)
            pygame.draw.rect(self.screen, COLORS['YELLOW'], bg_rect)
            pygame.draw.rect(self.screen, COLORS['BLACK'], bg_rect, 2)
            
            # 绘制文本
            self.screen.blit(text_surface, (60, 60))
    
    def run(self):
        """运行编辑器主循环"""
        running = True
        
        while running:
            # 处理事件
            running = self.handle_events()
            
            # 绘制界面
            self.draw()
            
            # 控制帧率
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()


def main():
    """主函数"""
    print("推箱子关卡编辑器")
    print("操作说明:")
    print("- 鼠标左键：放置选中的元素")
    print("- 鼠标右键：删除元素")
    print("- 数字键1-7：快速选择工具")
    print("- Ctrl+S：保存关卡")
    print("- Ctrl+N：新建关卡")
    print("- Ctrl+T：测试关卡")
    
    try:
        editor = SokobanLevelEditor()
        editor.run()
    except Exception as e:
        print(f"编辑器错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
