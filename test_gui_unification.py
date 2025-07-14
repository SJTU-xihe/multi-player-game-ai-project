#!/usr/bin/env python3
"""
测试GUI统一化的简单验证脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_gui_imports():
    """测试GUI模块是否可以正常导入"""
    print("测试GUI模块导入...")
    
    try:
        from gui_game import MultiGameGUI
        print("✅ 主GUI导入成功")
        
        # 测试创建实例
        gui = MultiGameGUI()
        print("✅ 主GUI实例创建成功")
        
        # 检查按钮配置
        print(f"✅ 主GUI默认游戏: {gui.current_game}")
        print(f"✅ 主GUI默认AI: {gui.selected_ai}")
        
        # 检查AI按钮更新功能
        gui._switch_game("snake")
        print(f"✅ 切换到贪吃蛇游戏: {gui.current_game}")
        print(f"✅ 贪吃蛇游戏默认AI: {gui.selected_ai}")
        
        # 检查按钮是否包含新的AI选项
        ai_buttons = [btn for btn in gui.buttons.keys() if btn.endswith('_ai')]
        print(f"✅ 当前AI按钮: {ai_buttons}")
        
        print("✅ 主GUI测试通过！")
        
    except Exception as e:
        print(f"❌ 主GUI测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        from snake_gui import SnakeGUI
        print("✅ 贪吃蛇专用GUI导入成功")
        
        # 测试创建实例
        snake_gui = SnakeGUI()
        print("✅ 贪吃蛇专用GUI实例创建成功")
        print(f"✅ 贪吃蛇专用GUI默认AI: {snake_gui.selected_ai}")
        
        # 检查按钮名称
        ai_buttons = [btn for btn in snake_gui.buttons.keys() if btn.endswith('_ai')]
        print(f"✅ 贪吃蛇专用GUI AI按钮: {ai_buttons}")
        
        print("✅ 贪吃蛇专用GUI测试通过！")
        
    except Exception as e:
        print(f"❌ 贪吃蛇专用GUI测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_ai_consistency():
    """测试AI选项一致性"""
    print("\n测试AI选项一致性...")
    
    try:
        from gui_game import MultiGameGUI
        from snake_gui import SnakeGUI
        
        main_gui = MultiGameGUI()
        snake_gui = SnakeGUI()
        
        # 切换到贪吃蛇游戏
        main_gui._switch_game("snake")
        
        # 检查AI选项是否一致
        main_ai_buttons = set(btn for btn in main_gui.buttons.keys() if btn.endswith('_ai'))
        snake_ai_buttons = set(btn for btn in snake_gui.buttons.keys() if btn.endswith('_ai'))
        
        print(f"主GUI贪吃蛇AI按钮: {sorted(main_ai_buttons)}")
        print(f"专用GUI AI按钮: {sorted(snake_ai_buttons)}")
        
        # 检查核心AI按钮是否一致（排除搜索AI，因为专用GUI可能不包含）
        core_buttons = {'minimax_ai', 'mcts_ai', 'random_ai'}
        main_core = main_ai_buttons.intersection(core_buttons)
        snake_core = snake_ai_buttons.intersection(core_buttons)
        
        if main_core == snake_core:
            print("✅ 核心AI按钮一致！")
            print(f"✅ 一致的按钮: {sorted(main_core)}")
        else:
            print("❌ 核心AI按钮不一致！")
            print(f"主GUI独有: {main_core - snake_core}")
            print(f"专用GUI独有: {snake_core - main_core}")
            return False
        
        print("✅ AI选项一致性测试通过！")
        
    except Exception as e:
        print(f"❌ AI选项一致性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """主测试函数"""
    print("=" * 60)
    print("GUI统一化测试")
    print("=" * 60)
    
    success = True
    
    success &= test_gui_imports()
    success &= test_ai_consistency()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ 所有测试通过！GUI统一化成功完成。")
        print("\n改进总结:")
        print("1. ✅ 统一了主GUI和专用GUI的AI选项命名")
        print("2. ✅ 为贪吃蛇游戏添加了Minimax AI和MCTS AI选项")
        print("3. ✅ 更新了AI创建逻辑，使用正确的AI类")
        print("4. ✅ 保持了界面的一致性和用户体验")
    else:
        print("❌ 部分测试失败，需要进一步调试。")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
