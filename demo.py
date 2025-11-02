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


class ScreenGenerator:

    def __init__(
        self,
        rows: int = 3,                                                            # 列數預設 3
        cols: int = 5,                                                            # 行數預設 5
        reel_strips: List[List[int]] = REELSTRIPS,                                # 做型別檢查用的必須是 2 維 list
        symbols: List[str] = SYMBOLS,                                             # 同上
        seed: Optional[int] = None,                                               # 同上
    ):
        self._valid() # 檢查合法性
        self.Rows = rows                                                          # 列數
        self.Cols = cols                                                          # 行數
        self.ScreenSize = rows * cols                                             # 盤面大小
        self.ReelStrips = np.asarray(reel_strips, dtype=np.uint8)                 # 輪帶表，轉成 np 陣列，型別用 uint8
        self.ReelLens = np.asarray([len(r) for r in reel_strips], dtype=np.int32) # 每條 reel 的長度
        self.Symbols = np.asarray(symbols,dtype=str)                              # 符號清單
        self.ScreenBuf = np.zeros(self.ScreenSize, dtype=np.uint8)                # 一次 spin 的輸出緩衝，初始 0 陣列狀態
        self.rng = np.random.Generator(np.random.PCG64(seed))                     # numpy 的亂數生成(帶種子固定結果)
        self._row_offsets = np.arange(rows, dtype=np.int64)                       # 第一列、第二列、第三列


    def gen_screen(self) -> np.ndarray:
        for i in range(self.Cols):
            reel = self.ReelStrips[i]                                   # 第 i 條輪帶
            L = reel.size                                               # 第 i 條輪帶的長度
            idx = self.rng.integers(L)                                  # 生成範圍內的整數隨機值
            take_idx = (idx + self._row_offsets) % L                    # 重複利用，不再配置
            start = i * self.Rows                                       # 因為是一維陣列要找存放的起始點
            self.ScreenBuf[start:start+self.Rows] = reel[take_idx]      # 存放
        return self.ScreenBuf

    # 檢查合法性
    def _valid(self) : 
        # 判斷 rows > 0
        # 判斷 cols > 0
        # REELSTRIPS[i].__len__() > rows
        # REELSTRIPS.__len__() == cols
        # REELSTRIPS[i][j] > 0 && REELSTRIPS[i][j] < len(symbols)
        return
    
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

def runner(rounds: int = 1_000_000, seed : int | None = None) :
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
    gener = ScreenGenerator(seed=seed)
    gener.gen_screen()
    print(gener.view_rows_cols())
    print(gener.as_symbol_names())

if __name__ == "__main__" :
    # runner()
    gen_screen_printer()

