from collections import deque
import numpy as np


class SampleBuffer(object):

    def __init__(self, frame, dim):
        super(SampleBuffer, self).__init__()
        self.frame = frame
        self.dim = dim
        self.queue = deque(maxlen=self.frame)
        for i in range(0, self.frame):
            self.queue.append(np.zeros(dim))

    def store(self, state):
        self.queue.append(state)
        A = list(self.queue)
        B = np.einsum('abc->bca', A)
        return B
