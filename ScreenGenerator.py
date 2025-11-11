from typing import Optional, List, Tuple
import numpy as np
from time import time

# -------------------------------------------------
# 基本設定資料
# -------------------------------------------------

REELSTRIPS = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # 第一輪
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # 第二輪
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # 第三輪
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # 第四輪
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # 第五輪
]

SYMBOLS = ["Z1", "C1", "W1", "H1", "H2", "H3", "H4",
           "L1", "L2", "L3", "L4", "L5"]  # 符號清單（index 0..11）

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
    [0, 0, 0, 0, 0],         # C1
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
        reel_strips: List[List[int]],
        symbols: List[str],
        lines: List[List[int]],
        pay_table: List[List[int]],
        rows: int = 3,
        cols: int = 5,
    ) -> None:
        self.rows = rows
        self.cols = cols

        # 每條輪帶使用一個一維 np.ndarray，包成 tuple 支援不同長度
        self.reel_strips: Tuple[np.ndarray, ...] = tuple(
            np.asarray(strip, dtype=np.uint8) for strip in reel_strips
        )
        # 其餘保持原本的 Python 結構，方便之後換機種
        self.symbols: List[str] = list(symbols)
        self.lines: List[List[int]] = [list(line) for line in lines]
        self.pay_table: List[List[int]] = [list(row) for row in pay_table]


DEFAULT_CONFIG = SlotConfig(
    reel_strips=REELSTRIPS,
    symbols=SYMBOLS,
    lines=LINES,
    pay_table=PAYTABLE,
)


# -------------------------------------------------
# 基底類別：SlotInit
# -------------------------------------------------

class SlotInit:
    """
    提供共用的基本屬性與合法性檢查：
    - Rows / Cols / ScreenSize
    - ReelStrips / ReelLens
    - Symbols / lines / PayTable / Bet
    """

    def __init__(self, config: SlotConfig = DEFAULT_CONFIG) -> None:
        self.Config = config

        self.Rows: int = config.rows
        self.Cols: int = config.cols
        self.ScreenSize: int = self.Rows * self.Cols

        self.ReelStrips: Tuple[np.ndarray, ...] = config.reel_strips
        self.ReelLens: np.ndarray = np.asarray(
            [len(r) for r in self.ReelStrips], dtype=np.int32
        )
        self.Symbols: np.ndarray = np.asarray(config.symbols, dtype=object)
        self.lines: np.ndarray = np.asarray(config.lines, dtype=np.uint8)
        self.PayTable: np.ndarray = np.asarray(config.pay_table, dtype=np.int64)
        self.Bet: int = 1000

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

        sym_len = len(self.Symbols)

        # 每條輪帶長度與符號索引範圍
        for i, reel in enumerate(self.ReelStrips):
            if reel.size < self.Rows:
                raise ValueError(f"第 {i} 條輪帶長度需 >= rows")
            if np.any((reel < 0) | (reel >= sym_len)):
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
        if self.PayTable.shape[0] != sym_len:
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
        seed: Optional[int] = None,
        config: SlotConfig = DEFAULT_CONFIG,
    ) -> None:
        super().__init__(config=config)
        self.ScreenBuf: np.ndarray = np.zeros(self.ScreenSize, dtype=np.uint8)
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self._row_offsets: np.ndarray = np.arange(self.Rows, dtype=np.int64)

    def gen_screen(self) -> np.ndarray:
        """
        生成一個隨機盤面（一維陣列，長度 = Rows * Cols）。
        """
        for col in range(self.Cols):
            reel = self.ReelStrips[col]
            length = reel.size
            idx = self.rng.integers(length)
            take_idx = (idx + self._row_offsets) % length
            start = col * self.Rows
            self.ScreenBuf[start:start + self.Rows] = reel[take_idx]
        return self.ScreenBuf

    def view_rows_cols(self) -> np.ndarray:
        """
        回傳形狀 (Rows, Cols) 的盤面視圖。
        """
        return self.ScreenBuf.reshape(self.Cols, self.Rows).T

    def as_symbol_names(self) -> np.ndarray:
        """
        回傳以符號名稱表示的盤面（Rows x Cols）。
        """
        names = self.Symbols
        return names[self.view_rows_cols()]


# -------------------------------------------------
# 判斷得分：SpinCalculator
# -------------------------------------------------

class SpinCalculator(SlotInit):
    """
    負責一個 spin 的數學計算：
    - 將一維盤面轉成每條線的符號 ID（trans_pay_line）
    - 判斷每條線是否中獎、算出符號 / 連線數 / 賠率（hit_check）
    - 計算盤面中的 C1 數量（count_c1）
    """

    def __init__(self, config: SlotConfig = DEFAULT_CONFIG) -> None:
        super().__init__(config=config)

        self.TotalWins: float = 0.0
        self.win: int = 0
        self.TotalBets: int = 0
        self.baseRtp: float = 0.0
        self.LineBuf: np.ndarray = np.zeros((self.Rows, self.Cols), dtype=np.uint8)
        self.transToLine: Optional[np.ndarray] = None
        self.base_idx: np.ndarray = np.arange(self.Cols, dtype=np.int64) * self.Rows

        # 最小連線數
        self.min_len: int = 3

        # Wild 的符號 ID
        wild_name = "W1"
        wild_matches = np.where(self.Symbols == wild_name)[0]
        self.wild_index: int = int(wild_matches[0]) if wild_matches.size > 0 else -1

        # C1 的符號 ID
        c1_matches = np.where(self.Symbols == "C1")[0]
        self.c1_index: int = int(c1_matches[0]) if c1_matches.size > 0 else -1

        # 不計分的符號：整列賠率都為 0 的 row
        zero_rows = np.all(self.PayTable == 0, axis=1)
        self.filter_ids: set[int] = set(np.where(zero_rows)[0])
        # Wild 即使賠率為 0 也不要排除（因為還要當替身用）
        if self.wild_index in self.filter_ids:
            self.filter_ids.remove(self.wild_index)

    def trans_pay_line(self, screen: np.ndarray) -> np.ndarray:
        """
        任務 2.1:
        根據當前盤面（一維 np array）與線獎組合，
        取得每條線上的實際數值（符號 ID）。
        """
        screen = np.asarray(screen, dtype=np.uint8)
        if screen.size != self.ScreenSize:
            raise ValueError(f"screen 長度應為 {self.ScreenSize}，實際為 {screen.size}")

        flat_idx = self.lines + self.base_idx  # (num_lines, Cols)
        self.transToLine = screen[flat_idx]
        return self.transToLine

    def hit_check(self, line_values: np.ndarray) -> list[tuple[int, int, int]]:
        """
        任務 2.2:
        給定所有線上的符號索引（transToLine），判斷每條線是否中獎。
        """
        line_values = np.asarray(line_values)
        if line_values.ndim == 1:
            line_values = line_values[np.newaxis, :]

        results: list[tuple[int, int, int]] = []

        for line in line_values:
            # 1. 找「得分符號 ID」 sym_id
            sym_id: int = -1
            for s in line:
                if s == self.wild_index:  # Wild 不當成得分符號
                    continue
                if int(s) in self.filter_ids:  # 例如 Z1, C1 等整列賠率為 0 的符號
                    sym_id = -1
                    break
                sym_id = int(s)  # 第一個合法非 Wild 的符號
                break

            # 2. 算得分符號連線數 sym_count
            if sym_id == -1:
                sym_count = 0
            else:
                sym_count = 1
                for s in line[1:]:
                    if s == sym_id or s == self.wild_index:
                        sym_count += 1
                    else:
                        break

            # 3. 算 Wild 連線數 wild_count
            wild_count = 0
            if self.wild_index != -1:
                for s in line:
                    if s == self.wild_index:
                        wild_count += 1
                    else:
                        break

            # 4. 計算兩種情況的賠率
            sym_pay = 0
            if sym_id != -1 and sym_count > 0:
                sym_pay = int(self.PayTable[sym_id, sym_count - 1])

            wild_pay = 0
            if self.wild_index != -1 and wild_count > 0:
                wild_pay = int(self.PayTable[self.wild_index, wild_count - 1])

            # 5. 判斷有沒有達到最小連線數
            if sym_count < self.min_len and wild_count < self.min_len:
                results.append((-1, 0, 0))
                continue

            # 6. 選擇「得分符號」或「純 Wild」
            if sym_pay >= wild_pay:
                win_sym_id = sym_id
                win_count = sym_count
                win_pay = sym_pay
            else:
                win_sym_id = self.wild_index
                win_count = wild_count
                win_pay = wild_pay

            results.append((win_sym_id, win_count, win_pay))

        return results

    def count_c1(self, screen: np.ndarray) -> int:
        """
        任務 2.3:
        計算一個盤面中出現多少個 C1。
        """
        if self.c1_index == -1:
            return 0

        arr = np.asarray(screen)
        if arr.ndim > 1:
            arr = arr.ravel()

        return int(np.count_nonzero(arr == self.c1_index))


# -------------------------------------------------
# 測試 / 執行
# -------------------------------------------------

def runner(rounds: int = 1_000_000, seed: Optional[int] = None) -> None:
    """
    進行多輪模擬：
    每一輪生成一個盤面、計算線獎與 C1 次數，最後統計 Base RTP。
    """
    config = DEFAULT_CONFIG
    generator = ScreenGenerator(seed=seed, config=config)
    calculator = SpinCalculator(config=config)

    print(f"running ScreenGenerator : gen {rounds:,d} screens")
    start = time()

    for i in range(1, rounds + 1):
        screen = generator.gen_screen()
        c1_count = calculator.count_c1(screen)
        line_values = calculator.trans_pay_line(screen)
        hits = calculator.hit_check(line_values)

        spin_pay = sum(pay for (_, _, pay) in hits)
        spin_win = spin_pay * calculator.Bet / 20  # 假設每線投注額為 Bet/20

        calculator.TotalWins += spin_win
        calculator.TotalBets += calculator.Bet

        if i == 1:
            print("---- First spin debug ----")
            print("Screen (ID):")
            print(generator.view_rows_cols())
            print("Screen (Symbols):")
            print(generator.as_symbol_names())
            print("C1 count:", c1_count)
            print("Line values (IDs):")
            print(line_values)
            print("Hit results (line, sym, count, pay):")
            for idx, (sym_id, cnt, pay) in enumerate(hits):
                if cnt > 0:
                    print(
                        f"  line {idx + 1}: sym_id={sym_id}, "
                        f"sym={calculator.Symbols[sym_id]}, "
                        f"cnt={cnt}, pay={pay}"
                    )
            print("--------------------------")

        if i % 100000 == 0:
            print(f"\r{i:,d} / {rounds:,d}", end="", flush=True)

    elapsed = time() - start
    print()
    print(f"used {elapsed:.2f} sec : gen {rounds:,d} screens")

    if calculator.TotalBets > 0:
        calculator.baseRtp = calculator.TotalWins / calculator.TotalBets
    else:
        calculator.baseRtp = 0.0

    print(f"TotalBet = {calculator.TotalBets}")
    print(f"TotalWin = {calculator.TotalWins}")
    print(f"Base RTP = {calculator.baseRtp:.6f}")


def gen_screen_printer(seed: Optional[int] = None) -> None:
    """
    生成一組盤面並將結果輸出到標準輸出視窗。
    """
    generator = ScreenGenerator(seed=seed)
    generator.gen_screen()
    print(generator.view_rows_cols())
    print(generator.as_symbol_names())


if __name__ == "__main__":
    runner(rounds=1_000_000, seed=42)
    # gen_screen_printer()
