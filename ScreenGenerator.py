from typing import Optional, List
import numpy as np
from time import time

# -------------------------------------------------
# 基本設定
# -------------------------------------------------

REELSTRIPS = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # 第一輪
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # 第二輪
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # 第三輪
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # 第四輪
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # 第五輪
]

SYMBOLS = ["Z1", "C1", "W1", "H1", "H2", "H3", "H4", "L1", "L2", "L3", "L4", "L5"]  # 符號清單（index 0..11）

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
    [0, 2, 2, 2, 0]
]

# 賠率表：每一 row 對應一個符號 ID，欄位是「1~5 連」的賠率
PAYTABLE = [
    [0, 0, 0, 0, 0],       # Z1
    [0, 0, 0, 0, 0],       # C1
    [0, 0, 100, 200, 300],       # W1 (Wild，這裡先設 0，看你之後調整)
    [0, 0, 10, 50, 200],   # H1
    [0, 0, 10, 50, 200],   # H2
    [0, 0, 10, 50, 200],   # H3
    [0, 0, 10, 50, 200],   # H4
    [0, 0, 5, 20, 100],    # L1
    [0, 0, 5, 20, 100],    # L2
    [0, 0, 5, 20, 100],    # L3
    [0, 0, 5, 20, 100],    # L4
    [0, 0, 5, 20, 100],    # L5
]


# -------------------------------------------------
# 基底類別：SlotInit
# -------------------------------------------------

class SlotInit:
    def __init__(
        self,
        rows: int = 3,                               # 列數預設 3
        cols: int = 5,                               # 行數預設 5
        reel_strips: List[List[int]] = REELSTRIPS,   # 輪帶表
        symbols: List[str] = SYMBOLS,                # 符號清單
        lines: List[List[int]] = LINES,              # 線獎組合
        pay_table=PAYTABLE                           # 賠率表
    ):
        """
        初始化基本參數與資料結構
        """
        self.Rows = rows
        self.Cols = cols
        self.ScreenSize = rows * cols

        # 轉成 numpy 陣列
        self.ReelStrips = np.asarray(reel_strips, dtype=np.uint8)          # shape: (Cols, reel_len)
        self.ReelLens = np.asarray([len(r) for r in reel_strips],
                                   dtype=np.int32)                         # 每條輪帶長度
        self.Symbols = np.asarray(symbols, dtype=object)                   # 符號名稱表
        self.lines = np.asarray(lines, dtype=np.uint8)                     # 線獎定義
        self.PayTable = np.asarray(pay_table, dtype=np.int64)              # 賠率表
        self.Bet = 1000                                                    # 單線下注（或之後你要改）

        self._valid()                                                      # 檢查合法性

    def _valid(self) -> None:
        """
        檢查初始化參數是否合法（簡單版）。
        """
        if self.Rows <= 0:
            raise ValueError("rows 必須 > 0")
        if self.Cols <= 0:
            raise ValueError("cols 必須 > 0")

        # 檢查輪帶條數
        if self.ReelStrips.shape[0] != self.Cols:
            raise ValueError("reel_strips 條數必須等於 cols")

        sym_len = len(self.Symbols)

        for i, reel in enumerate(self.ReelStrips):
            if reel.size < self.Rows:
                raise ValueError(f"第 {i} 條輪帶長度需 >= rows")
            # 檢查符號索引是否在合法範圍 [0, sym_len-1]
            if np.any((reel < 0) | (reel >= sym_len)):
                raise ValueError(f"第 {i} 條輪帶中有非法符號索引")


# -------------------------------------------------
# 產生盤面：ScreenGenerator
# -------------------------------------------------

class ScreenGenerator(SlotInit):

    def __init__(
        self,
        seed: Optional[int] = None,  # 隨機種子
    ):
        super().__init__()           # 用預設的 reels/lines/paytable
        self.ScreenBuf = np.zeros(self.ScreenSize, dtype=np.uint8)   # 一次 spin 結果緩存
        self.rng = np.random.Generator(np.random.PCG64(seed))        # numpy 亂數生成器
        self._row_offsets = np.arange(self.Rows, dtype=np.int64)     # [0, 1, ..., Rows-1]

    def gen_screen(self) -> np.ndarray:
        """
        生成一個隨機盤面（一維陣列長度 = Rows * Cols）
        """
        for i in range(self.Cols):
            reel = self.ReelStrips[i]                       # 第 i 條輪帶
            L = reel.size                                   # 輪帶長度
            idx = self.rng.integers(L)                      # 隨機起始位置
            take_idx = (idx + self._row_offsets) % L        # 連續 Rows 個位置
            start = i * self.Rows
            self.ScreenBuf[start:start + self.Rows] = reel[take_idx]
        return self.ScreenBuf

    def view_rows_cols(self) -> np.ndarray:
        """
        返回: 形狀 (Rows, Cols) 的視圖（視覺化較直觀）。
        """
        return self.ScreenBuf.reshape(self.Cols, self.Rows).T

    def as_symbol_names(self) -> np.ndarray:
        """
        返回: 以符號名稱矩陣（Rows x Cols）回傳，方便除錯或輸出。
        """
        names = self.Symbols
        return names[self.view_rows_cols()]


# -------------------------------------------------
# 判斷得分：SpinCalculator
# -------------------------------------------------

class SpinCalculator(SlotInit):

    def __init__(self):
        """
        根據當前盤面與線獎組合，做中獎判斷。
        """
        super().__init__()  # 使用同一套設定

        self.TotalWins = 0                      # 總贏分（金額）
        self.win = 0                            # 單次贏分
        self.TotalBets = 0                      # 總下注
        self.baseRtp = 0                        # Base game RTP
        self.LineBuf = np.zeros((3, 5), dtype=np.uint8)
        self.transToLine: np.ndarray | None = None
        self.base_idx = np.arange(self.Cols, dtype=np.int64) * self.Rows  # 每一軸起始 index

        # 任務 2.2 相關設定
        self.min_len = 3  # 最小連線數

        # Wild 的「符號名稱」與 ID
        self.wild_name = "W1"
        matches = np.where(self.Symbols == self.wild_name)[0]
        self.wild_index = int(matches[0]) if matches.size > 0 else -1

        # ---- C1 的 ID ----
        c1_matches = np.where(self.Symbols == "C1")[0]
        self.c1_index = int(c1_matches[0]) if c1_matches.size > 0 else -1

        # 不計分的符號：整列賠率都為 0 的 row
        zero_rows = np.all(self.PayTable == 0, axis=1)
        self.filter_ids: set[int] = set(np.where(zero_rows)[0])
        # Wild 即使賠率為 0 也不要排除（因為還要當替身用）
        if self.wild_index in self.filter_ids:
            self.filter_ids.remove(self.wild_index)

    def transPayLine(self, screen: np.ndarray) -> np.ndarray:
        """
        任務 2.1:
        根據當前盤面（一維 np array）與線獎組合，取得每條線上的實際數值（符號 ID）。

        Args:
            screen: shape = (Rows*Cols,) 的一維 np.ndarray，內容為符號 ID。

        Returns:
            shape = (num_lines, Cols) 的二維陣列，每列代表一條線上的符號 ID。
        """
        screen = np.asarray(screen, dtype=np.uint8)
        if screen.size != self.ScreenSize:
            raise ValueError(f"screen 長度應為 {self.ScreenSize}，實際為 {screen.size}")

        # self.lines: (num_lines, Cols)，元素是 row index (0,1,2)
        # flat index = col * Rows + row
        flat_idx = self.lines + self.base_idx  # broadcasting: (num_lines, Cols)
        self.transToLine = screen[flat_idx]
        return self.transToLine

    def hitCheck(self, line_values: np.ndarray) -> List[tuple[int, int, int]]:
        """
        任務 2.2:
        給定所有線上的符號索引（transToLine），判斷每條線是否中獎。

        Args:
            line_values: shape = (num_lines, Cols) 的 np.ndarray，
                         每個元素是「符號 ID」（對應 self.Symbols 的 index）。

        Returns:
            List[tuple]: [(win_sym_id, win_count, win_pay), ...]
                win_sym_id : 中獎符號 ID（-1 表示沒中）
                win_count  : 連線數（0 表示沒中）
                win_pay    : 該線賠率（尚未乘下注金額）
        """
        line_values = np.asarray(line_values)
        if line_values.ndim == 1:
            line_values = line_values[np.newaxis, :]

        results: List[tuple[int, int, int]] = []

        for line in line_values:
            # line 例如: [3, 3, 2, 3, 7] （符號 ID）

            # ---------- 1. 找「得分符號 ID」 sym_id ----------

            
            sym_id: int = -1
            for s in line:
                if s == self.wild_index:     # Wild 不當成得分符號
                    continue
                if int(s) in self.filter_ids:  # 例如 Z1, C1 等整列賠率為 0 的符號
                    sym_id = -1
                    break
                sym_id = int(s)             # 第一個合法非 Wild 的符號
                break

            # ---------- 2. 算得分符號連線數 sym_count ----------
            if sym_id == -1:
                sym_count = 0
            else:
                sym_count = 1
                for s in line[1:]:
                    if s == sym_id or s == self.wild_index:
                        sym_count += 1
                    else:
                        break

            # ---------- 3. 算 Wild 連線數 wild_count ----------
            wild_count = 0
            if self.wild_index != -1:
                for s in line:
                    if s == self.wild_index:
                        wild_count += 1
                    else:
                        break

            # ---------- 4. 計算兩種情況的賠率 ----------
            sym_pay = 0
            if sym_id != -1 and sym_count > 0:
                sym_pay = int(self.PayTable[sym_id, sym_count - 1])

            wild_pay = 0
            if self.wild_index != -1 and wild_count > 0:
                wild_pay = int(self.PayTable[self.wild_index, wild_count - 1])

            # ---------- 5. 判斷有沒有達到最小連線數 ----------
            if sym_count < self.min_len and wild_count < self.min_len:
                results.append((-1, 0, 0))   # 沒中獎
                continue

            # ---------- 6. 選擇「得分符號」或「純 Wild」 ----------
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
        任務2.3 :
        計算一個盤面中出現多少個 C1。

        Args:
            screen: 一維或二維 np.ndarray，元素為符號 ID
                    - 一維: shape = (Rows*Cols,)
                    - 二維: shape = (Rows, Cols)

        Returns:
            int: 盤面中 C1 的個數。
        """
        if self.c1_index == -1:
            return 0  # 沒有 C1 這個符號就直接回 0

        arr = np.asarray(screen)
        if arr.ndim > 1:
            arr = arr.ravel()  # 攤平成一維

        return int(np.count_nonzero(arr == self.c1_index))





# -------------------------------------------------
# 測試 / 執行
# -------------------------------------------------

def runner(rounds: int = 1_000_000, seed: int | None = None):
    """
    生成多組盤面並檢查每組盤面的分數。
    """
    gener = ScreenGenerator(seed=seed)
    Calc = SpinCalculator()


    print(f"running ScreenGenerator : gen {rounds:,d} screens")
    start = time()

    for i in range(1, rounds + 1):
        screen = gener.gen_screen()                   # 一維盤面
        c1_count = Calc.count_c1(screen)            # 計算 C1 數量          
        line_values = Calc.transPayLine(screen)     # 線上符號 ID 矩陣
        hits = Calc.hitCheck(line_values)           # 每條線中獎結果

        # 計算這一 spin 的贏分（先用賠率 * Bet）
        spin_pay = sum(pay for (_, _, pay) in hits)
        spin_win = spin_pay * Calc.Bet

        Calc.TotalWins += spin_win # 更新贏分
        Calc.TotalBets += Calc.Bet # 更新下注總分

        # 第一把印出細節看一下
        if i == 1:
            print("---- First spin debug ----")
            print("Screen (ID):")
            print(gener.view_rows_cols())
            print("Screen (Symbols):")
            print(gener.as_symbol_names())
            print("Line values (IDs):")
            print(line_values)
            print("Hit results (line, sym, count, pay):")
            for idx, (sym_id, cnt, pay) in enumerate(hits):
                if cnt > 0:
                    print(
                        f"  line {idx + 1}: sym_id={sym_id}, "
                        f"sym={Calc.Symbols[sym_id]}, "
                        f"cnt={cnt}, pay={pay}"
                    )
            print("--------------------------")

        if i % 100000 == 0:
            print(f"\r{i:,d} / {rounds:,d}", end="", flush=True)

    elapsed = time() - start
    print()
    print(f"used {elapsed:.2f} sec : gen {rounds:,d} screens")

    if Calc.TotalBets > 0:
        Calc.baseRtp = Calc.TotalWins / Calc.TotalBets
    else:
        Calc.baseRtp = 0.0

    print(f"TotalBet = {Calc.TotalBets}")
    print(f"TotalWin = {Calc.TotalWins}")
    print(f"Base RTP = {Calc.baseRtp:.6f}")


def gen_screen_printer(seed: int | None = None):
    """
    生成一組盤面並將結果輸出到標準輸出視窗。
    """
    gener = ScreenGenerator(seed=seed)
    gener.gen_screen()
    print(gener.view_rows_cols())
    print(gener.as_symbol_names())


if __name__ == "__main__":
    runner()
    # gen_screen_printer()
