#!/usr/bin/env python3
"""
快速测试修复后的观察值处理
"""

def test_observation_fix():
    """测试观察值格式修复"""
    print("测试观察值格式修复...")
    
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        import numpy as np
        from games.gomoku.gomoku_env import GomokuEnv
        from examples.simple_ai_examples import RuleBasedGomokuBot
        
        # 创建环境和智能体
        env = GomokuEnv(board_size=15)
        agent = RuleBasedGomokuBot("TestAgent", player_id=1)
        
        # 重置环境获取观察值
        observation, info = env.reset()
        print(f"观察值类型: {type(observation)}")
        print(f"观察值形状: {observation.shape if hasattr(observation, 'shape') else 'N/A'}")
        
        # 测试智能体是否能处理观察值
        valid_actions = env.get_valid_actions()
        print(f"有效动作数量: {len(valid_actions)}")
        
        # 尝试获取动作
        action = agent.get_action(observation, env)
        print(f"智能体选择的动作: {action}")
        
        if action in valid_actions:
            print("✓ 成功！智能体正确处理了观察值格式")
            return True
        else:
            print("✗ 失败：智能体返回了无效动作")
            return False
            
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_observation_fix()
