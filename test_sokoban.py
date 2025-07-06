#!/usr/bin/env python3
"""
推箱子游戏测试
验证游戏逻辑和AI功能
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_sokoban_basic():
    """测试推箱子基本功能"""
    print("=== 测试推箱子基本功能 ===")
    
    try:
        from games.sokoban import SokobanGame, SokobanEnv
        print("✓ 成功导入推箱子游戏模块")
        
        # 创建游戏实例
        game = SokobanGame(level_id=1, game_mode='competitive')
        print("✓ 成功创建游戏实例")
        
        # 重置游戏
        state = game.reset()
        print("✓ 成功重置游戏")
        print(f"  游戏板大小: {game.height}x{game.width}")
        print(f"  目标数量: {len(game.targets)}")
        print(f"  箱子数量: {len(game.boxes)}")
        
        # 测试有效动作
        valid_actions = game.get_valid_actions()
        print(f"✓ 有效动作: {valid_actions}")
        
        # 测试移动
        if valid_actions:
            old_pos = game.player1_pos
            result = game.step(valid_actions[0])
            new_pos = game.player1_pos
            print(f"✓ 执行动作 {valid_actions[0]}: {old_pos} -> {new_pos}")
            print(f"  奖励: {result[1]}")
            print(f"  结束: {result[2]}")
        
        # 渲染游戏
        print("\n游戏状态:")
        print(game.render())
        
        return True
        
    except Exception as e:
        print(f"✗ 基本功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sokoban_env():
    """测试推箱子环境"""
    print("\n=== 测试推箱子环境 ===")
    
    try:
        from games.sokoban import SokobanEnv
        
        # 创建环境
        env = SokobanEnv(level_id=1, game_mode='competitive')
        print("✓ 成功创建环境")
        
        # 重置环境
        observation, info = env.reset()
        print("✓ 成功重置环境")
        print(f"  观察空间形状: {observation['board'].shape}")
        print(f"  当前玩家: {observation['current_player']}")
        
        # 测试步骤
        valid_actions = env.get_legal_actions()
        if valid_actions:
            obs, reward, terminated, truncated, info = env.step(valid_actions[0])
            print(f"✓ 执行步骤: 动作={valid_actions[0]}, 奖励={reward}")
            print(f"  终止: {terminated}, 截断: {truncated}")
        
        # 测试可用关卡
        levels = env.get_available_levels()
        print(f"✓ 可用关卡数量: {len(levels)}")
        for level in levels[:3]:  # 显示前3个关卡
            print(f"  关卡{level['id']}: {level['name']} ({level['difficulty']})")
        
        return True
        
    except Exception as e:
        print(f"✗ 环境测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sokoban_ai():
    """测试推箱子AI"""
    print("\n=== 测试推箱子AI ===")
    
    try:
        from games.sokoban import SokobanEnv
        from agents.ai_bots.sokoban_ai import SokobanAI, SimpleSokobanAI
        
        # 创建环境和AI
        env = SokobanEnv(level_id=1, game_mode='competitive')
        smart_ai = SokobanAI(name="Smart AI", player_id=1)
        simple_ai = SimpleSokobanAI(name="Simple AI", player_id=2)
        print("✓ 成功创建AI智能体")
        
        # 重置环境
        observation, info = env.reset()
        
        # 测试AI决策
        smart_action = smart_ai.get_action(observation, env)
        simple_action = simple_ai.get_action(observation, env)
        
        print(f"✓ Smart AI 动作: {smart_action}")
        print(f"✓ Simple AI 动作: {simple_action}")
        
        # 测试AI游戏
        print("\n模拟AI对战几步:")
        for step in range(min(5, 10)):  # 最多10步
            if env.is_terminal():
                break
                
            current_player = env.get_current_player()
            if current_player == 1:
                action = smart_ai.get_action(observation, env)
                agent_name = "Smart AI"
            else:
                action = simple_ai.get_action(observation, env)
                agent_name = "Simple AI"
            
            if action:
                observation, reward, terminated, truncated, info = env.step(action)
                print(f"  步骤{step+1}: {agent_name} 执行 {action}, 奖励={reward:.2f}")
                
                if terminated or truncated:
                    winner = env.get_winner()
                    print(f"  游戏结束! 获胜者: {winner}")
                    break
            else:
                print(f"  {agent_name} 无法找到有效动作")
                break
        
        return True
        
    except Exception as e:
        print(f"✗ AI测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gui_import():
    """测试GUI导入"""
    print("\n=== 测试GUI导入 ===")
    
    try:
        # 这里只测试导入，不运行GUI
        import sokoban_gui
        print("✓ 成功导入推箱子GUI模块")
        
        # 检查关键类是否存在
        if hasattr(sokoban_gui, 'SokobanGUI'):
            print("✓ SokobanGUI类存在")
        else:
            print("✗ SokobanGUI类不存在")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ GUI导入测试失败: {e}")
        return False

def run_all_tests():
    """运行所有测试"""
    print("推箱子游戏测试开始...")
    print("=" * 50)
    
    tests = [
        test_sokoban_basic,
        test_sokoban_env,
        test_sokoban_ai,
        test_gui_import
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print("测试失败!")
        except Exception as e:
            print(f"测试出现异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"测试完成: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过! 推箱子游戏已成功实现!")
        print("\n使用方法:")
        print("1. 运行 'python sokoban_gui.py' 启动图形界面")
        print("2. 使用方向键控制玩家移动")
        print("3. 将棕色箱子推到粉色目标点")
        print("4. 选择不同的AI对手和游戏模式")
    else:
        print("❌ 部分测试失败，请检查错误信息")
    
    return passed == total

if __name__ == "__main__":
    # 设置环境
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'  # 隐藏pygame提示
    
    success = run_all_tests()
    
    if success:
        print("\n是否运行推箱子游戏GUI? (y/n): ", end="")
        try:
            choice = input().lower().strip()
            if choice in ['y', 'yes', '是']:
                print("启动推箱子游戏GUI...")
                import sokoban_gui
                sokoban_gui.main()
        except (KeyboardInterrupt, EOFError):
            print("\n程序退出")
    
    sys.exit(0 if success else 1)
