from typing import Optional, List
import numpy as np
from time import time

# REELSTRIPS = [
#     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], # 第一輪
#     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], # 第二輪
#     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], # 第三輪
#     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], # 第四輪
#     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], # 第五輪
# ]
REELSTRIPS = [
    [2,2,2,2,2,2,2,2,2,2,7, 8, 9, 10, 11], # 第一輪
    [2,2,2,2,2,2,2,2,2,2,7, 8, 9, 10, 11], # 第二輪
    [2,2,2,2,2,2,2,2,2,2,7, 8, 9, 10, 11], # 第三輪
    [2,2,2,2,2,2,2,2,2,2,7, 8, 9, 10, 11], # 第四輪
    [2,2,2,2,2,2,2,2,2,2,7, 8, 9, 10, 11], # 第五輪
] # 測試 Wild 連線用

SYMBOLS = ["Z1", "C1", "W1", "H1", "H2", "H3", "H4", "L1", "L2", "L3", "L4", "L5"] # 符號清單

# 線獎組合
LINES =  [
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



PAYTABLE = [[0,0,0,0,0],  # Z1 賠率表
            [0,0,0,0,0],  # C1 賠率表
            [0,0,300,600,600],  # W1 賠率表
            [0,0,10,50,200],  # H1 賠率 表
            [0,0,10,50,200],  # H2 賠率表
            [0,0,10,50,200],  # H3 賠率表
            [0,0,10,50,200],  # H4 賠率表
            [0,0,5,20,100],  # L1 賠率表
            [0,0,5,20,100],  # L2 賠率表
            [0,0,5,20,100],  # L3 賠率表
            [0,0,5,20,100],  # L4 賠率表
            [0,0,5,20,100],  # L5 賠率表
] # 賠率表

class SlotInit:
    def __init__(
        self,
        rows: int = 3,                               # 列數預設 3
        cols: int = 5,                               # 行數預設 5
        reel_strips: List[List[int]] = REELSTRIPS,   # 輪帶表
        symbols: List[str] = SYMBOLS,                # 符號清單
        lines: List[List[int]] = LINES,              # 線獎組合
        payTable: dict = PAYTABLE                    # 賠率表
    ):
        """
        初始化
        """
        self._valid()                                                                   # 檢查合法性
        self.Rows = rows                                                                # 列數
        self.Cols = cols                                                                # 行數
        self.ScreenSize = rows * cols                                                   # 盤面大小
        self.ReelStrips = np.asarray(reel_strips, dtype=np.uint8)                       # 輪帶表
        self.ReelLens = np.asarray([len(r) for r in reel_strips], dtype=np.int32)       # 每條 reel 的長度
        self.Symbols = np.asarray(symbols, dtype=object)                                # 符號清單
        self.lines = np.asarray(lines, dtype=np.uint8)                                  # 線獎組合
        self.PayTable = payTable                                                        # 賠率表
        self.Bet = 1000                                                                 # 下注金額

    def _valid(self) -> None:
        """
        檢查初始化參數是否合法（簡單版）。
        """
        if self.Rows <= 0:
            raise ValueError("rows 必須 > 0")
        if self.Cols <= 0:
            raise ValueError("cols 必須 > 0")

        # 檢查輪帶條數
        if len(self.ReelStrips) != self.Cols:
            raise ValueError("reel_strips 條數必須等於 cols")

        sym_len = len(self.Symbols)

        for i, reel in enumerate(self.ReelStrips):
            if len(reel) < self.Rows:
                raise ValueError(f"第 {i} 條輪帶長度需 >= rows")
            # 檢查符號索引是否在合法範圍 [0, sym_len-1]
            if np.any((reel < 0) | (reel >= sym_len)):
                raise ValueError(f"第 {i} 條輪帶中有非法符號索引")

class ScreenGenerator(SlotInit):

    def __init__(
        self,
        seed: Optional[int] = None,                                               # 隨機種子
    ):
        super().__init__()                                                        # 繼承父類別初始化  
        self._valid()                                                             # 檢查數值合法性                                                 
        self.ScreenBuf = np.zeros(self.ScreenSize, dtype=np.uint8)                # 一次 spin 結果緩存，初始 0 陣列狀態
        self.rng = np.random.Generator(np.random.PCG64(seed))                     # numpy 的亂數生成(帶種子固定結果)
        print(f"Seed used: {seed}")
        self._row_offsets = np.arange(self.Rows, dtype=np.int64)                  # [0, 1, 2, ..., Rows-1]


    def gen_screen(self) -> np.ndarray:
        for i in range(self.Cols):
            reel = self.ReelStrips[i]                                             # 第 i 條輪帶
            L = reel.size                                                         # 第 i 條輪帶的長度
            idx = self.rng.integers(L)                                            # 生成範圍內的整數隨機值
            take_idx = (idx + self._row_offsets) % L                              # 重複利用，不再配置
            start = i * self.Rows                                                 # 因為是一維陣列要找存放的起始點
            self.ScreenBuf[start:start+self.Rows] = reel[take_idx]                # 存放
        return self.ScreenBuf

    def _valid(self) :
        """
        檢查初始化參數是否合法。
        """
        pass
    
    def view_rows_cols(self) -> np.ndarray:
        """
        返回: 形狀 (Rows, Cols) 的視圖（一般視覺化較直觀）。
        """
        return self.ScreenBuf.reshape(self.Cols, self.Rows).T
    
    def as_symbol_names(self) -> np.ndarray:
        """
        返回: 以符號名稱矩陣（Rows x Cols）回傳，方便除錯或輸出。
        """
        names = np.asarray(self.Symbols, dtype=object)
        return names[self.view_rows_cols()]

# 
class ScreenViewer(SlotInit):

    def __init__(self, screen: Optional[np.ndarray] = None):
        """
        根据當前盤面與線獎組合，做中獎判斷。
        """
        super().__init__()                                                        # 繼承父類別初始化  
        self._valid()                                                             # 檢查數值合法性
        self.TotalWins = 0                                                        # 總贏分
        self.win = 0                                                              # 單次贏分
        self.TotalBets = 0                                                        # 總下注分
        self.baseRtp = None                                                       # base game RTP
        self.LineBuf = np.zeros((3, 5), dtype=np.uint8)                           # 目前用不到，先保留
        self.transToLine: np.ndarray | None = None                                # 轉換後的線獎組合緩存
        self.base_idx = np.arange(self.Cols, dtype=np.int64) * self.Rows          # 每一列的起始索引

        # ---- 這邊是任務 2.2 會用到的參數 ----
        self.min_len: int = 3                       # 最小連線數，對應 mini_len（$DN$3）
        self.wild_name: str = "W1"                  # Wild 的符號名稱（你現在是 W1）

        # PayTable 轉成 numpy，方便取值
        self.PayTable = np.asarray(self.PayTable, dtype=np.int64)

        # Wild 的 row index
        matches = np.where(self.Symbols == self.wild_name)[0]
        self.wild_index: int | None = int(matches[0]) if matches.size > 0 else None

        # filter_sym：預設為「整排賠率都為 0 的符號」，視為不能當得分符號
        zero_rows = np.all(self.PayTable == 0, axis=1)
        self.filter_symbols: set[str] = set(self.Symbols[zero_rows])
        # 但不要把 Wild 自己也排除
        if self.wild_name in self.filter_symbols:
            self.filter_symbols.remove(self.wild_name)

    def _valid(self):
        """
        檢查參數是否合法。（目前先不做額外檢查）
        """
        return
    
    def transPayLine(self, screen: np.ndarray) -> np.ndarray:
        """
        任務 2.1:
        根據當前盤面（一維 np array）與線獎組合，取得每條線上的實際數值。
        """
        screen = np.asarray(screen, dtype=np.uint8)

        flat_idx = self.lines + self.base_idx  # shape = (num_lines, Cols)
        self.transToLine = screen[flat_idx]    # shape = (num_lines, Cols)

        return self.transToLine
    
    def hitCheck(self, line_values: np.ndarray) -> List[tuple[str, int, int]]:
        """
        任務 2.2:
        給定所有線上的符號索引（transToLine），判斷每條線是否中獎。

        Args:
            line_values: shape = (num_lines, Cols) 的 np.ndarray，
                         每個元素是符號索引（對應 self.Symbols）。

        Returns:
            List[tuple]: [(win_sym, win_count, win_pay), ...]
                win_sym   : 中獎符號名稱（或 "" 表示沒中）
                win_count : 連線數（0 表示沒中）
                win_pay   : 該線賠率（尚未乘下注金額）
        """
        # line_values = np.asarray(line_values, dtype=np.uint8) # 確保是 np.ndarray

        if line_values.ndim == 1:
            line_values = line_values[np.newaxis, :]

        results: List[tuple[str, int, int]] = [] # results 會放 (符號, 連線數, 這個符號的連線數對應的賠率)

        # 
        for line in line_values:
            # 這條線上的「符號名稱」陣列，例如 ["H1","H1","W1","H1","L1"]
            names = self.Symbols[line]
            # print("Line symbols:", names)

            # ---------- 1. 找「得分符號」 sym ----------
            # 從左往右找第一個不是 Wild 的符號
            sym: str | None = None
            for n in names:
                if n != self.wild_name:
                    sym = n
                    break

            # 若 sym 在 filter_symbols 裡（例如 Z1, C1），視為不能當得分符號
            if sym is not None and sym in self.filter_symbols:
                sym = None

            # ---------- 2. 算得分符號連線數 sym_count ----------
            # 規則：第一軸一定算 1 格（如果有 sym），後面遇到 sym 或 Wild 就 +1，直到斷掉
            if sym is None:
                sym_count = 0
            else:
                sym_count = 1
                for n in names[1:]:
                    if n == sym or n == self.wild_name:
                        sym_count += 1
                    else:
                        break

            # ---------- 3. 算 Wild 連線數 wild_count ----------
            # 規則：從最左起連續多少個 Wild
            wild_count = 0
            for n in names:
                if n == self.wild_name:
                    wild_count += 1
                else:
                    break

            # ---------- 4. 計算兩種情況的賠率 ----------
            sym_pay = 0
            if sym is not None and sym_count > 0:
                sym_idx = int(np.where(self.Symbols == sym)[0][0])
                # PayTable 每列 5 個，代表 1~5 連，所以 index = count-1
                sym_pay = int(self.PayTable[sym_idx, sym_count - 1])

            wild_pay = 0
            if wild_count > 0 and self.wild_index is not None:
                wild_pay = int(self.PayTable[self.wild_index, wild_count - 1])

            # ---------- 5. 判斷有沒有達到最小連線數 ----------
            if sym_count < self.min_len and wild_count < self.min_len:
                # 兩種都沒達到最小連線數 => 沒中
                results.append(("", 0, 0))
                continue

            # ---------- 6. 選擇要用「得分符號」還是「純 Wild」來計算 ----------
            # 對應 Excel:
            #   win_sym   = IF(sym_pay>=wild_pay, sym, "Wild")
            #   win_count = IF(sym_pay>=wild_pay, sym_count, wild_count)
            #   win_pay   = IF(sym_pay>=wild_pay, sym_pay, wild_pay)
            if sym_pay >= wild_pay:
                win_sym  = sym if sym is not None else ""  # sym 可能是 None
                win_count = sym_count
                win_pay   = sym_pay
            else:
                win_sym  = self.wild_name
                win_count = wild_count
                win_pay   = wild_pay

            results.append((win_sym, win_count, win_pay))

        return results


    

def runner(rounds: int = 1_000_000, seed : int | None = None) :
    gener = ScreenGenerator(seed=seed)
    viewer = ScreenViewer()

    print(f"running ScreenGenerator : gen {rounds:,d} screens")
    start = time() 

    for i in range(1,rounds+1) :
        screen = gener.gen_screen()
        line_values = viewer.transPayLine(screen) # 轉換線獎組合
        hits = viewer.hitCheck(line_values)

        if i == 1:  # 只印第 1 組盤面看一下
            print("screen (index) =")
            print(gener.view_rows_cols())
            print("lines values (index) =")
            print(line_values)

            print("hit results (line, sym, count, pay) =")
            for idx, (sym, cnt, pay) in enumerate(hits):
                print(f"  line {idx+1}: {sym} x{cnt}, pay = {pay}")

        if i % 100000 == 0 :
            print(f"\r{i:,d} / {rounds:,d}", end="", flush=True)

    elapsed = time()-start
    print()
    print(f'used {elapsed:.2f} sec : gen {rounds:,d} screens')




def gen_screen_printer(seed: int | None = None) :
    """
    生成一組盤面並將結果輸出到標準輸出視窗。

    會建立一個 ScreenGenerator 實例並呼叫 gen_screen() 產生盤面，
    接著列印盤面的列／欄視圖以及對應的符號名稱。

    Args:
        seed (int | None, optional): 隨機種子。提供相同的 seed 時會得到相同的盤面。
            預設為 None，表示使用系統隨機種子。

    Returns:
        None
    """
    gener = ScreenGenerator(seed=seed)
    gener.gen_screen()
    print(gener.view_rows_cols())
    print(gener.as_symbol_names())

if __name__ == "__main__" :
    runner()
    # gen_screen_printer()

