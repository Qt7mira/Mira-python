import random


class MaxHeap(object):

    def __init__(self):
        self.sz = 0
        self.limit = 1 << 20
        self.data = []

    def set_limit(self, limit):
        self.limit = limit
        if self.limit < 1:
            self.limit = 1

    def insert(self, x):
        self.sz += 1
        self.data.append(x)
        p = self.sz - 1
        while p > 0:
            np = p >> 1
            if self.data[np] < x:
                self.data[p] = self.data[np]
                p = np
            else:
                break
        self.data[p] = x

    def down(self, p):
        x = self.data[p]
        while p * 2 < self.sz:
            l = p << 1
            r = l + 1
            if r < self.sz and self.data[r] > self.data[l]:
                l = r
            if self.data[l] <= x:
                break
            self.data[p] = self.data[l]
            p = l
        self.data[p] = x

    def add(self, x):
        if self.sz == self.limit:
            if self.data[0] > x:
                self.data[0] = x
                self.down(0)
        else:
            self.insert(x)

    def show(self):
        print(self.data)


if __name__ == "__main__":
    heap = MaxHeap()
    heap.set_limit(5)
    #find the minimal 5 numbers
    n = 20
    for i in range(n):
        val = random.random() * 10
        heap.add(val)
        print("val: ", val, ";  ", heap.data)
