# 有输入输出指标，不需要相关性分析，自动选择权重
# 只有DEA才能给出改进建议


import numpy as np
import pandas as pd


class topsis:
    a = None  # Matrix
    w = None  # Weight matrix
    r = None  # Normalisation matrix
    m = None  # Number of rows
    n = None  # Number of columns
    aw = []  # worst alternative
    ab = []  # best alternative
    diw = None
    dib = None
    siw = None
    sib = None

    # Return a numpy array with float items
    def floater(self, a):
        ax = []
        for i in a:
            try:
                ix = []
                for j in i:
                    ix.append(float(j))
                    # jj = "%.4f" % j
                    # ix.append(jj)
            except:
                ix = float(i)
                pass
            ax.append(ix)
        return np.array(ax)

    # Step 1
    def __init__(self, a, w):
        self.a = self.floater(a)
        self.m = len(a)
        self.n = len(a[0])
        self.w = self.floater(w)
        self.w = self.w / sum(self.w)

    # Step 2
    def step2(self):
        self.r = self.a
        for j in range(self.n):
            nm = sum(self.a[:, j]**2) ** 0.5
            for i in range(self.m):
                self.r[i, j] = self.a[i,j]/nm

    # Step 3
    def step3(self):
        self.t = self.r * self.w

    # Step 4
    def step4(self):
        for i in range(self.n):
            self.aw.append(min(self.t[:, i]))
            self.ab.append(max(self.t[:, i]))

    # Step 5,6
    def step5(self):
        diw_list = []
        dib_list = []
        siw_list = []
        for i in range(self.m):
            list_diw = []
            list_dib = []
            for j in range(self.n):
                list_diw.append((self.t[i,j] - self.aw[j])**2)
                list_dib.append((self.t[i,j] - self.ab[j])**2)
            single_diw = sum(list_diw)**0.5
            single_dib = sum(list_dib)**0.5
            single_siw = single_diw / (single_diw + single_dib)
            diw_list.append(single_diw)
            dib_list.append(single_dib)
            siw_list.append(single_siw)



            # Step 5
    def step5(self):
        self.diw = (self.t - self.aw) ** 2
        self.dib = (self.t - self.ab) ** 2
        print("diw & dib:")
        print(self.diw)
        print(self.dib)
        """for j in range(self.n):
            self.diw[:,j]=(self.diw[:,j]-self.aw[j])**2
            self.dib[:,j]=(self.dib[:,j]-self.ab[j])**2
        print self.diw"""
        self.dw = []
        self.db = []
        for j in range(self.m):
            self.dw.append(sum(self.diw[j, :]) ** 0.5)
            self.db.append(sum(self.dib[j, :]) ** 0.5)
        # print(self.dw)
        self.dw = np.array(self.dw)
        self.db = np.array(self.db)
        print(self.dw)
        print(self.db)

    # print self.db

    # Step 6
    def step6(self):
        np.seterr(all='ignore')
        self.siw = self.dw / (self.dw + self.db)
        # print self.siw
        x = 0
        m = None
        for i in range(self.m):
            print(self.siw[i])
            if self.siw[i] > self.m or self.m is None:
                m = self.siw[i]
                x = i
        print('Choice', x + 1, 'is the best')

    def calc(self):
        self.step2()
        self.step3()
        self.step4()
        self.step55()
        # self.step6()

if __name__ == '__main__':
    # get data and weight
    path = "C:\\Users\\MaTao\\Desktop\\mba\\DMD\\dmd_data.xlsx"

    sheetname_data = "data1"
    sheetname_weight = "weight1"

    df_data = pd.read_excel(path, sheetname=sheetname_data)
    df_weight = pd.read_excel(path, sheetname=sheetname_weight)
    # print(df_data)
    # print(df_weight)

    matrix_data = df_data.as_matrix(columns=None)
    matrix_weight = df_weight.as_matrix(columns=None)

    # weightlength = [1,2,3,4,5,6,7]

    tp = topsis(matrix_data, matrix_weight)
    tp.calc()

