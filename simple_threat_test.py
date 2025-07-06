#!/usr/bin/env python3

# 直接测试威胁检测逻辑（无需导入复杂模块）
import numpy as np

def test_threat_detection_simple():
    """直接测试威胁检测函数"""
    print("直接测试活三威胁检测逻辑...")
    
    # 模拟关键函数
    def line_has_live_three(line, player, target_idx):
        """简化版活三检测"""
        # 标准化
        normalized = []
        for cell in line:
            if cell == player:
                normalized.append(1)
            elif cell == 0:
                normalized.append(0)
            else:
                normalized.append(-1)
        
        # 检查标准活三 [0,1,1,1,0]
        for i in range(len(normalized) - 4):
            if normalized[i:i+5] == [0, 1, 1, 1, 0]:
                if target_idx == i or target_idx == i + 4:
                    return True
        return False
    
    def check_direction_threat(board, row, col, dr, dc, opponent_id):
        """简化版方向威胁检测"""
        board_size = board.shape[0]
        
        # 检查5长度的线段
        for start_offset in range(-4, 1):
            start_r = row + start_offset * dr
            start_c = col + start_offset * dc
            
            if (0 <= start_r < board_size and 0 <= start_c < board_size and
                0 <= start_r + 4*dr < board_size and 0 <= start_c + 4*dc < board_size):
                
                line = []
                for i in range(5):
                    r = start_r + i * dr
                    c = start_c + i * dc
                    line.append(board[r, c])
                
                target_idx = -start_offset  # 目标位置在线段中的索引
                if target_idx >= 0 and target_idx < 5:
                    if line_has_live_three(line, opponent_id, target_idx):
                        return True
        return False
    
    def blocks_opponent_threat(board, row, col, opponent_id):
        """简化版威胁阻挡检测"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            if check_direction_threat(board, row, col, dr, dc, opponent_id):
                return True
        return False
    
    # 测试场景：水平活三
    print("\n测试场景：水平活三")
    board = np.zeros((15, 15), dtype=int)
    board[7, 5] = 1  # X
    board[7, 6] = 1  # X
    board[7, 7] = 1  # X
    
    print("棋盘第7行: ", end="")
    for j in range(3, 10):
        if board[7, j] == 1:
            print("X", end="")
        else:
            print(".", end="")
    print(" (位置3-9)")
    
    # 测试防守位置
    test_positions = [(7, 4), (7, 8), (8, 8)]  # 正确防守位置和错误位置
    
    for pos in test_positions:
        row, col = pos
        if board[row, col] == 0:  # 只测试空位
            threat_blocked = blocks_opponent_threat(board, row, col, 1)
            print(f"位置{pos}能阻挡威胁: {threat_blocked}")
    
    # 检查具体的威胁检测
    print("\n详细检查位置(7,4):")
    line = [board[7, j] for j in range(4, 9)]  # [0,1,1,1,0]
    print(f"线段: {line}")
    threat = line_has_live_three(line, 1, 0)  # 目标位置在索引0
    print(f"包含活三威胁: {threat}")
    
    print("\n详细检查位置(7,8):")
    line = [board[7, j] for j in range(4, 9)]  # [0,1,1,1,0]
    print(f"线段: {line}")
    threat = line_has_live_three(line, 1, 4)  # 目标位置在索引4
    print(f"包含活三威胁: {threat}")

if __name__ == "__main__":
    test_threat_detection_simple()
