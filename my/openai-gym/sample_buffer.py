from collections import deque
from numpy import zeros, reshape


class SampleBuffer(object):

    def __init__(self, frame, dim):
        super(SampleBuffer, self).__init__()
        self.frame = frame
        self.dim = dim
        self.queue = deque(maxlen=self.frame)
        for i in range(0, self.frame):
            self.queue.append(zeros([dim], dtype=float))

    def store(self, state):
        self.queue.append(state)
        return reshape(list(self.queue), [self.frame * self.dim])
