from math import sqrt


class Score(object):
    def __init__(self, Nhit, Nref, Nf):
        self.HR = round(Nhit / Nref * 100, 5)
        self.OS = round((Nf / Nref - 1) * 100, 5)
        self.PRC = round(Nhit / Nf, 5)
        self.RCL = round(Nhit / Nref, 5)
        self.F = round(2 * self.PRC * self.RCL / (self.PRC + self.RCL), 5)
        r1 = sqrt((100 - self.HR) ** 2 + self.OS ** 2)
        r2 = (-self.OS + self.HR - 100) / sqrt(2)
        self.R = round(1 - (abs(r1) + abs(r2)) / 200, 5)