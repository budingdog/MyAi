class Rewarder(object):

    def __init__(self):
        self.last_info = None

    def get_reward(self, reward, done, info):
        res_result = 0
        if done == False:
            if reward > 0:
                res_result = 1.0
            elif self.last_info != None and self.last_info['ale.lives'] - info['ale.lives'] > 0:
                res_result = -1.0
            else:
                res_result = 0
        else:
            if info['ale.lives'] > 0:
                res_result = 1.0
            else:
                res_result = -1.0
        self.last_info = info
        return res_result