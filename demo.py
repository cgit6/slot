import random
import numpy as np

totaltimes = int(input("请输入模拟次数: "))  # 输入模拟次数

BET = 1000  # 下注金额

# 符号列表
Symbols = ["K", "Q", "J"]

# 赔付倍数配置: [五连, 四连, 三连]
SymbolKOdds = [30, 20, 5]  # 图标 K 赔付倍数配置
SymbolQOdds = [15, 10, 5]  # 图标 Q 赔付倍数配置
SymbolJOdds = [15, 10, 5]  # 图标 J 赔付倍数配置

# 概率配置（万分比）
SymbolRateK = [3000, 3000, 3000, 3000, 3000]  # 图标 K 概率配置
SymbolRateQ = [3000, 3000, 3000, 3000, 3000]  # 图标 Q 概率配置
SymbolRateJ = [4000, 4000, 4000, 4000, 4000]  # 图标 J 概率配置

SymbolOdds = [SymbolKOdds, SymbolQOdds, SymbolJOdds]

AllRTP = []

for _ in range(10):  # 开始模拟 10 次数据
    times = totaltimes
    TotalBet = 0
    WIN = 0

    while times > 0:
        TotalBet += BET

        # 生成 3x5 盘面
        LandingSymbols = []
        for _row in range(3):
            line = []
            for col in range(5):
                RandOdds = random.randint(1, 10000)
                if RandOdds <= SymbolRateK[col]:
                    line.append("K")
                elif RandOdds <= SymbolRateK[col] + SymbolRateQ[col]:
                    line.append("Q")
                else:
                    line.append("J")
            LandingSymbols.append(line)

        # 中奖线配置（3 条水平线：上中下）
        PayLine1Setting = [0, 0, 0, 0, 0]
        PayLine2Setting = [1, 1, 1, 1, 1]
        PayLine3Setting = [2, 2, 2, 2, 2]
        PayLinesSetting = [PayLine1Setting, PayLine2Setting, PayLine3Setting]

        # 根据中奖线配置，取出每条线上的实际符号
        PayLines = []
        for setting in PayLinesSetting:
            line_symbols = []
            for col, row_idx in enumerate(setting):
                line_symbols.append(LandingSymbols[row_idx][col])
            PayLines.append(line_symbols)

        # 比较中奖线和中奖组合，结算赢钱数额
        for sym_index, sym in enumerate(Symbols):
            CombinationFive = [sym] * 5
            CombinationFour = [sym] * 4
            CombinationThree = [sym] * 3

            for line in PayLines:
                # 5 连
                if line == CombinationFive:
                    WIN += SymbolOdds[sym_index][0] * BET / len(PayLines)
                # 4 连（从第 1 轮开始算）
                elif line[:4] == CombinationFour:
                    WIN += SymbolOdds[sym_index][1] * BET / len(PayLines)
                # 3 连
                elif line[:3] == CombinationThree:
                    WIN += SymbolOdds[sym_index][2] * BET / len(PayLines)

        times -= 1

    RTP = WIN / TotalBet if TotalBet > 0 else 0
    AllRTP.append(RTP)
    print("RTP =", RTP)

print("平均RTP, 方差, 标准差 = ",
      np.mean(AllRTP),
      np.var(AllRTP),
      np.sqrt(np.var(AllRTP)))
