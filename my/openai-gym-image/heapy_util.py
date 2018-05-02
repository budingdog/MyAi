from guppy import hpy


class HeapyUtil(object):

    def __init__(self, path):
        self.h = hpy()
        self.path = path

    def dump(self):
        self.h.setrelheap()
        with open(self.path, 'w') as f:
            heap = self.h.heap();
            f.write(str(heap))
