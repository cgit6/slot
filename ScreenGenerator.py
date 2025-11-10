from typing import Optional, List
import numpy as np
from time import time

REELSTRIPS = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], # 第一輪
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], # 第二輪
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], # 第三輪
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], # 第四輪
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], # 第五輪
]

SYMBOLS = ["Z1", "C1", "W1", "H1", "H2", "H3", "H4", "L1", "L2", "L3", "L4", "L5"] # 符號清單


LINES = [[]] # 線獎組合


PAYTABLE = {} # 賠率表

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
        self.lines = lines                                                              # 線獎組合
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
        self._valid()                                                             # 檢查合法性                                                 
        self.ScreenBuf = np.zeros(self.ScreenSize, dtype=np.uint8)                # 一次 spin 的輸出緩衝，初始 0 陣列狀態
        self.rng = np.random.Generator(np.random.PCG64(seed))                     # numpy 的亂數生成(帶種子固定結果)
        self._row_offsets = np.arange(self.Rows, dtype=np.int64)                  # 第一列、第二列、第三列


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

def runner(rounds: int = 1_000_000, seed : int | None = None) :
    "執行"
    # SlotInit() # 先初始化一次，檢查參數合法性
    gener = ScreenGenerator(seed=seed)
    print(f"running ScreenGenerator : gen {rounds:,d} screens")
    start = time()
    for i in range(1,rounds+1) :
        gener.gen_screen()
        if i%100000 == 0 :
            print(f"\r{i:,d} / {rounds:,d}",end="",flush=True)
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

