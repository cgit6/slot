import random
import numpy as np


class ScreenGenerator :
    def __init__ (self):
        self.rows = 3 # 盤面列數
        self.cols = 5 # 盤面行數 
        self.reel_strips = np.array(3,dtype=int) # 一組輪帶表


        self.rand_seed = random.seed(123456789) # 隨機種子


    def test_rand_seed(self):
        print(random.random()) # 在物件中固定


if __name__ == "__main__":
    ScreenGen = ScreenGenerator()
    ScreenGen.test_rand_seed() # 測試隨機結果
