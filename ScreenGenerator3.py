from typing import Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
from time import time
import math

# -------------------------------------------------
# 基本設定資料（這裡當作預設機台設定）
# -------------------------------------------------

REELSTRIPS = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # 第一輪
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # 第二輪
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # 第三輪
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # 第四輪
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # 第五輪
]

SYMBOLS = [
    "Z1", "C1", "W1",
    "H1", "H2", "H3", "H4",
    "L1", "L2", "L3", "L4", "L5",
]  # index 0..11

# 線獎組合：每一條線是「列索引」組成的長度 5 陣列
LINES = [
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0],
    [2, 2, 2, 2, 2],
    [0, 1, 2, 1, 0],
    [2, 1, 0, 1, 2],
    [1, 0, 0, 0, 1],
    [1, 2, 2, 2, 1],
    [0, 0, 1, 2, 2],
    [2, 2, 1, 0, 0],
    [1, 0, 1, 2, 1],
    [1, 2, 1, 0, 1],
    [0, 1, 1, 1, 0],
    [2, 1, 1, 1, 2],
    [0, 1, 0, 1, 0],
    [2, 1, 2, 1, 2],
    [1, 1, 0, 1, 1],
    [1, 1, 2, 1, 1],
    [0, 0, 2, 0, 0],
    [2, 2, 0, 2, 2],
    [0, 2, 2, 2, 0],
]

# 賠率表：每一 row 對應一個符號 ID，欄位是「1~5 連」的賠率
PAYTABLE = [
    [0, 0, 0, 0, 0],         # Z1
    [0, 0, 0, 0, 0],         # C1 (Scatter)
    [0, 0, 100, 200, 300],   # W1 (Wild)
    [0, 0, 10, 50, 200],     # H1
    [0, 0, 10, 50, 200],     # H2
    [0, 0, 10, 50, 200],     # H3
    [0, 0, 10, 50, 200],     # H4
    [0, 0, 5, 20, 100],      # L1
    [0, 0, 5, 20, 100],      # L2
    [0, 0, 5, 20, 100],      # L3
    [0, 0, 5, 20, 100],      # L4
    [0, 0, 5, 20, 100],      # L5
]


# -------------------------------------------------
# 遊戲靜態設定：SlotConfig
# -------------------------------------------------

class SlotConfig:
    """
    遊戲靜態設定：輪帶、符號、線獎與賠率表。
    """

    def __init__(
        self,
        reelStrips: List[List[int]],   # 各輪輪帶定義 (Cols 條，每條長度可不同)
        symbols: List[str],            # 符號名稱列表
        lines: List[List[int]],        # 線獎組合 (row index pattern)
        payTable: List[List[int]],     # 賠率表 (symbol x 1~5 連賠率)
        rows: int = 3,                 # 列數
        cols: int = 5,                 # 行數(輪數)
    ) -> None:
        self.rows = rows
        self.cols = cols

        # 每條輪帶使用一個一維 np.ndarray，包成 tuple 支援不同長度
        self.reelStrips: Tuple[np.ndarray, ...] = tuple(
            np.asarray(strip, dtype=np.uint8) for strip in reelStrips
        )
        # 其餘保持原本的 Python 結構
        self.symbols: List[str] = list(symbols)
        self.lines: List[List[int]] = [list(line) for line in lines]
        self.payTable: List[List[int]] = [list(row) for row in payTable]


DEFAULT_CONFIG = SlotConfig(
    reelStrips=REELSTRIPS,
    symbols=SYMBOLS,
    lines=LINES,
    payTable=PAYTABLE,
)


# -------------------------------------------------
# 基底類別：SlotInit
# -------------------------------------------------

class SlotInit:
    """
    共用的基本屬性與合法性檢查：

    - Rows / Cols / ScreenSize
    - ReelStrips / ReelLens
    - Symbols / lines / PayTable / Bet
    """

    def __init__(self, config: SlotConfig = DEFAULT_CONFIG) -> None:
        self.Config = config

        self.Rows: int = config.rows
        self.Cols: int = config.cols
        self.ScreenSize: int = self.Rows * self.Cols

        self.ReelStrips: Tuple[np.ndarray, ...] = config.reelStrips
        self.ReelLens: np.ndarray = np.asarray(
            [len(r) for r in self.ReelStrips], dtype=np.int32
        )
        self.Symbols: np.ndarray = np.asarray(config.symbols, dtype=object)
        self.lines: np.ndarray = np.asarray(config.lines, dtype=np.uint8)
        self.PayTable: np.ndarray = np.asarray(config.payTable, dtype=np.int64)
        self.Bet: int = 1000  # 預設單線下注

        self._valid()

    def _valid(self) -> None:
        """
        檢查初始化參數是否合法。
        """
        # Rows / Cols
        if self.Rows <= 0:
            raise ValueError("rows 必須 > 0")
        if self.Cols <= 0:
            raise ValueError("cols 必須 > 0")

        # 輪帶數量
        if len(self.ReelStrips) != self.Cols:
            raise ValueError("reel_strips 條數必須等於 Cols")

        symLen = len(self.Symbols)

        # 每條輪帶長度與符號索引範圍
        for i, reel in enumerate(self.ReelStrips):
            if reel.size < self.Rows:
                raise ValueError(f"第 {i} 條輪帶長度需 >= rows")
            if np.any((reel < 0) | (reel >= symLen)):
                raise ValueError(f"第 {i} 條輪帶中有非法符號索引")

        # 線獎陣列檢查
        if self.lines.ndim != 2:
            raise ValueError("lines 必須是二維陣列 (num_lines, Cols)")
        if self.lines.shape[1] != self.Cols:
            raise ValueError("每條線的長度必須等於 Cols")
        if np.any((self.lines < 0) | (self.lines >= self.Rows)):
            raise ValueError("線獎中的 row index 超出盤面列數")

        # 賠率表檢查
        if self.PayTable.ndim != 2:
            raise ValueError("PayTable 必須是二維陣列")
        if self.PayTable.shape[0] != symLen:
            raise ValueError("PayTable 列數必須等於符號數量")
        if self.PayTable.shape[1] != 5:
            raise ValueError("目前假設 1~5 連，PayTable 欄數必須是 5")


# -------------------------------------------------
# 產生盤面：ScreenGenerator
# -------------------------------------------------

class ScreenGenerator(SlotInit):
    """
    負責生成隨機盤面（spin）。
    """

    def __init__(
        self,
        seed: Optional[int] = None,             # 隨機種子
        config: SlotConfig = DEFAULT_CONFIG,    # 機台靜態設定
    ) -> None:
        super().__init__(config=config)

        self.ScreenBuf: np.ndarray = np.zeros(self.ScreenSize, dtype=np.uint8)
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self._rowOffsets: np.ndarray = np.arange(self.Rows, dtype=np.int64)

    def genScreen(self) -> np.ndarray:
        """
        生成一個隨機盤面（一維陣列，長度 = Rows * Cols）。
        """
        for col in range(self.Cols):
            reel = self.ReelStrips[col]
            length = reel.size
            idx = self.rng.integers(length)
            takeIdx = (idx + self._rowOffsets) % length
            start = col * self.Rows
            self.ScreenBuf[start:start + self.Rows] = reel[takeIdx]
        return self.ScreenBuf

    def viewRowsCols(self) -> np.ndarray:
        """
        回傳形狀 (Rows, Cols) 的盤面視圖。
        """
        return self.ScreenBuf.reshape(self.Cols, self.Rows).T

    def asSymbolNames(self) -> np.ndarray:
        """
        回傳以符號名稱表示的盤面（Rows x Cols）。
        """
        names = self.Symbols
        return names[self.viewRowsCols()]


# -------------------------------------------------
# 計算結果資料結構
# -------------------------------------------------

@dataclass
class LineResult:
    """
    單條線的得分結果。
    """
    lineIndex: int   # 線號（0-based）
    symbolId: int    # 中獎符號 ID（-1 表示沒中）
    count: int       # 連線數
    pay: int         # 該線賠率（尚未乘 line bet）


@dataclass
class ScreenResult:
    """
    一次盤面計算的結果。
    """
    screen: np.ndarray              # 盤面（一維 ID 陣列）
    c1Count: int                    # 盤面中 C1 出現次數
    lineResults: list[LineResult]   # 每條線的結果
    totalLinePay: int               # 各線賠率合計 (sum pay)
    totalWin: int                   # 最終贏分（已乘 lineBet）


# -------------------------------------------------
# 判斷得分：SpinCalculator
# -------------------------------------------------

class SpinCalculator(SlotInit):
    """
    `SpinCalculator` 在「給定一個盤面（screen）」的前提下，完成整個得分計算流程。

    對外只暴露一個主要介面：
        calcScreen(screen, lineBet) -> ScreenResult
    """

    def __init__(
        self,
        config: SlotConfig = DEFAULT_CONFIG,   # 機台靜態設定
    ) -> None:
        super().__init__(config=config)

        # 模擬統計用欄位（可選）
        self.TotalWins: int = 0
        self.TotalBets: int = 0
        self.baseRtp: float = 0.0

        # 每一軸在一維盤面中的起始 index：col * Rows
        self.baseIndex: np.ndarray = np.arange(self.Cols, dtype=np.int64) * self.Rows

        # 最小連線數
        self.minLen: int = 3

        # Wild / C1 的符號 ID（這裡直接用 index，效率較好）
        self.wildIndex: int = 2  # "W1"
        self.c1Index: int = 1    # "C1"

        # 不計分的符號：整列賠率都為 0 的 row
        zeroRows = np.all(self.PayTable == 0, axis=1)
        self.filterIds: set[int] = set(np.where(zeroRows)[0])
        # Wild 即使賠率為 0 也不要排除（因為還要當替身用）
        if self.wildIndex in self.filterIds:
            self.filterIds.remove(self.wildIndex)

    # ---------- Debug 用：從一條線的 pattern 直接取得該線上的符號 ID ----------

    def _getLineSymbols(self, screen: np.ndarray, linePattern: np.ndarray) -> np.ndarray:
        """
        給定一維盤面與某條線的 row pattern，
        回傳該條線上依序的符號 ID（一維長度 Cols）。

        ※ 只在 runner 第一把做 debug 用，計算邏輯本身不依賴「線表」。
        """
        lineSymbols = np.empty(self.Cols, dtype=np.uint8)
        for col in range(self.Cols):
            rowIndex = int(linePattern[col])
            flatIndex = self.baseIndex[col] + rowIndex
            lineSymbols[col] = screen[flatIndex]
        return lineSymbols

    # ---------- 小工具：計算賠率與最小連線判斷 ----------

    def _calcSymPay(self, symId: int, symCount: int) -> int:
        """
        得分符號的賠率 symPay。
        """
        if symId == -1 or symCount <= 0:
            return 0
        return int(self.PayTable[symId, symCount - 1])

    def _calcWildPay(self, wildCount: int) -> int:
        """
        純 Wild 連線的賠率 wildPay。
        """
        if self.wildIndex == -1 or wildCount <= 0:
            return 0
        return int(self.PayTable[self.wildIndex, wildCount - 1])

    def _belowMinLen(self, symCount: int, wildCount: int) -> bool:
        """
        判斷是否兩種連線都沒達到最小連線數。
        """
        return symCount < self.minLen and wildCount < self.minLen

    def _chooseWinner(
        self,
        symId: int,
        symCount: int,
        symPay: int,
        wildCount: int,
        wildPay: int,
    ) -> tuple[int, int, int]:
        """
        在得分符號與純 Wild 之間選擇賠率較高的組合。

        回傳 (winSymId, winCount, winPay)。
        """
        if symPay >= wildPay:
            return symId, symCount, symPay
        else:
            return self.wildIndex, wildCount, wildPay

    # ---------- 單條線：單一迴圈計算 wild / sym 連線與賠率 ----------

    def _hitCheckLineSinglePass(
        self,
        screen: np.ndarray,
        linePattern: np.ndarray,
    ) -> tuple[int, int, int]:
        """
        針對「一條線」，用單一迴圈同時計算：
        - wildCount：開頭連續 Wild 的數量
        - symId / symCount：得分符號與其連線數（含 Wild 代替）
        """
        wildCount    = 0
        wildContinue = True

        symId        = -1
        symCount     = 0
        pendingWilds = 0
        symStarted   = False

        for col in range(self.Cols):
            # 1. 從一維盤面抓出這一軸上該線的那一格
            rowIndex  = int(linePattern[col])
            flatIndex = self.baseIndex[col] + rowIndex
            symbol    = int(screen[flatIndex])

            # 2. 更新 wildCount（開頭那串連續 Wild）
            if wildContinue and symbol == self.wildIndex:
                wildCount += 1
            else:
                wildContinue = False

            # 3. 更新 symId / symCount
            if not symStarted:
                # sym 還沒決定的階段
                if symbol == self.wildIndex:
                    # 先累積在 pendingWilds，之後若遇到 sym 會算進去
                    pendingWilds += 1
                    continue

                # 第一個非 Wild
                if symbol in self.filterIds:
                    # 不計分符號，例如 Z1 / C1，這條線只能依靠前面 Wild
                    break

                # 合法得分符號出現 → symId 決定
                symId      = symbol
                symStarted = True
                symCount   = pendingWilds + 1  # 前面的 Wild 全算進連線
            else:
                # symId 已決定，延伸連線
                if symbol == symId or symbol == self.wildIndex:
                    symCount += 1
                else:
                    break

        # 4. 算出 sym / wild 各自的賠率
        symPay  = self._calcSymPay(symId, symCount)
        wildPay = self._calcWildPay(wildCount)

        # 5. 若兩種都未達最小連線數 → 沒中
        if self._belowMinLen(symCount, wildCount):
            return -1, 0, 0

        # 6. 在得分符號與純 Wild 之間選一個賠率較高的
        return self._chooseWinner(symId, symCount, symPay, wildCount, wildPay)

    def _countC1(self, screen: np.ndarray) -> int:
        """
        統計盤面中的 C1 數量。
        """
        if self.c1Index == -1:
            return 0

        arr = np.asarray(screen)
        if arr.ndim > 1:
            arr = arr.ravel()

        return int(np.count_nonzero(arr == self.c1Index))

    # ---------- 對外主要介面：一次計算一個 screen ----------

    def calcScreen(self, screen: np.ndarray, lineBet: Optional[int] = None) -> ScreenResult:
        """
        外部呼叫入口。
        """
        arr = np.asarray(screen, dtype=np.uint8)
        if arr.size != self.ScreenSize:
            raise ValueError(f"screen 長度應為 {self.ScreenSize}，實際為 {arr.size}")

        if lineBet is None:
            lineBet = self.Bet

        # 統計 C1
        c1Count = self._countC1(arr)

        lineResults: list[LineResult] = []
        totalLinePay = 0

        # 不建立線表矩陣，逐條線直接用「單一迴圈」從 screen 計算
        for lineIndex, linePattern in enumerate(self.lines):
            symId, count, pay = self._hitCheckLineSinglePass(arr, linePattern)

            lineResults.append(
                LineResult(
                    lineIndex=lineIndex,
                    symbolId=symId,
                    count=count,
                    pay=pay,
                )
            )
            totalLinePay += pay

        totalWin = int(totalLinePay * lineBet)

        return ScreenResult(
            screen=arr.copy(),
            c1Count=c1Count,
            lineResults=lineResults,
            totalLinePay=totalLinePay,
            totalWin=totalWin,
        )


# -------------------------------------------------
# Game 物件：把 Generator + Calculator 包起來
# -------------------------------------------------

class SlotGame:
    """
    對外的「遊戲」物件，提供一個 Spin() 介面：
        result = game.Spin()
    """

    def __init__(
        self,
        config: SlotConfig = DEFAULT_CONFIG,        # 機台靜態設定
        seed: Optional[int] = None,                # 隨機種子
        lineBet: int = 1000,                       # 單線下注
    ) -> None:
        self.Config = config
        self.LineBet = lineBet
        self.Generator = ScreenGenerator(seed=seed, config=config)
        self.Calculator = SpinCalculator(config=config)
        self.NumLines = self.Calculator.lines.shape[0]

    def Spin(self) -> ScreenResult:
        """
        執行一次 spin：產生盤面 + 計算得分。
        """
        screen = self.Generator.genScreen()
        result = self.Calculator.calcScreen(screen, lineBet=self.LineBet)
        return result


# -------------------------------------------------
# Stat：統計物件（期望值、波動、標準差）
# -------------------------------------------------

class Stat:
    """
    負責統計模擬結果：
    - 每一把的「回報率」 r_i = win_i / bet_i
    - RTP = sum(win) / sum(bet) = 平均 r_i
    - Std = r_i 的標準差
    - CV  = Std / RTP
    """

    def __init__(self) -> None:
        self.SpinCount: int = 0
        self.TotalBet: float = 0.0
        self.TotalWin: float = 0.0
        self._sumReturn: float = 0.0      # sum(r_i)
        self._sumReturnSq: float = 0.0    # sum(r_i^2)

    def Record(self, spinWin: float, spinBet: float) -> None:
        """
        記錄一把的結果。
        """
        if spinBet <= 0:
            return

        r = spinWin / spinBet  # 單把回報率

        self.SpinCount += 1
        self.TotalBet += spinBet
        self.TotalWin += spinWin
        self._sumReturn += r
        self._sumReturnSq += r * r

    @property
    def Rtp(self) -> float:
        if self.TotalBet <= 0:
            return 0.0
        return self.TotalWin / self.TotalBet

    @property
    def MeanReturn(self) -> float:
        if self.SpinCount == 0:
            return 0.0
        return self._sumReturn / self.SpinCount

    @property
    def Std(self) -> float:
        """
        回報率 r_i 的標準差（population std）。
        """
        n = self.SpinCount
        if n == 0:
            return 0.0
        mean = self._sumReturn / n
        meanSq = self._sumReturnSq / n
        var = max(meanSq - mean * mean, 0.0)
        return math.sqrt(var)

    @property
    def Cv(self) -> float:
        """
        變異係數 = Std / MeanReturn。
        """
        m = self.MeanReturn
        if m == 0:
            return 0.0
        return self.Std / m


# -------------------------------------------------
# Simulator：模擬器物件
# -------------------------------------------------

class SlotSimulator:
    """
    負責跑多輪模擬：
        result = game.Spin()
        stat.Record(result.totalWin, totalBet)
    """

    def __init__(
        self,
        game: SlotGame,          # 遊戲物件
        rounds: int,             # 模擬局數
    ) -> None:
        self.Game = game
        self.Rounds = rounds

    def Run(self, stat: Optional[Stat] = None, debugFirstSpin: bool = True) -> Stat:
        """
        執行模擬，回傳 Stat。
        """
        if stat is None:
            stat = Stat()

        numLines = self.Game.NumLines
        lineBet = self.Game.LineBet

        print(f"running simulator : spin {self.Rounds:,d} rounds")
        start = time()

        for i in range(1, self.Rounds + 1):
            result = self.Game.Spin()
            spinBet = lineBet * numLines
            stat.Record(result.totalWin, spinBet)

            if debugFirstSpin and i == 1:
                self._debugFirstSpin(result)

            if i % 100000 == 0:
                print(f"\r{i:,d} / {self.Rounds:,d}", end="", flush=True)

        elapsed = time() - start
        print()
        print(f"used {elapsed:.2f} sec : spin {self.Rounds:,d} rounds")

        return stat

    def _debugFirstSpin(self, result: ScreenResult) -> None:
        """
        第一把做詳細 debug 用。
        """
        game = self.Game
        gen = game.Generator
        calc = game.Calculator

        print("---- First spin debug ----")
        print("Screen (ID):")
        print(gen.viewRowsCols())
        print("Screen (Symbols):")
        print(gen.asSymbolNames())
        print("C1 count:", result.c1Count)

        numLines = calc.lines.shape[0]
        debugLines = np.empty((numLines, calc.Cols), dtype=np.uint8)
        for li, linePattern in enumerate(calc.lines):
            debugLines[li] = calc._getLineSymbols(result.screen, linePattern)

        print("Line values (IDs):")
        print(debugLines)
        print("Hit results (line, sym, count, pay):")
        for r in result.lineResults:
            if r.count > 0 and r.symbolId >= 0:
                print(
                    f"  line {r.lineIndex + 1}: "
                    f"symId={r.symbolId}, "
                    f"sym={calc.Symbols[r.symbolId]}, "
                    f"cnt={r.count}, pay={r.pay}"
                )
        print("--------------------------")


# -------------------------------------------------
# 測試 / 模擬執行
# -------------------------------------------------

def runner(rounds: int = 1_000_000, seed: Optional[int] = None) -> None:
    """
    高階入口：
    - 建立 Game
    - 建立 Simulator
    - 建立 Stat
    - 跑模擬後輸出 RTP / Std / CV
    """
    config = DEFAULT_CONFIG

    game = SlotGame(
        config=config,
        seed=seed,
        lineBet=1000,
    )

    simulator = SlotSimulator(
        game=game,
        rounds=rounds,
    )

    stat = Stat()
    stat = simulator.Run(stat=stat, debugFirstSpin=True)

    print(f"TotalBet = {stat.TotalBet:.0f}")
    print(f"TotalWin = {stat.TotalWin:.0f}")
    print(f"RTP      = {stat.Rtp:.6f}")
    print(f"Std(r)   = {stat.Std:.6f}")
    print(f"CV       = {stat.Cv:.6f}")


def genScreenPrinter(seed: Optional[int] = None) -> None:
    """
    生成一組盤面並將結果輸出到標準輸出視窗。
    """
    generator = ScreenGenerator(seed=seed)
    generator.genScreen()
    print(generator.viewRowsCols())
    print(generator.asSymbolNames())


if __name__ == "__main__":
    runner(rounds=1_000_000, seed=42)
    # genScreenPrinter()
