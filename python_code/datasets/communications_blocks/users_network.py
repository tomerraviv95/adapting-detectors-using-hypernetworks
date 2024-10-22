from python_code import conf
from python_code.utils.constants import Phase, MAX_USERS

N_USER = 14

class UsersNetwork:
    def __init__(self, phase: Phase, blocks_num: int):
        if phase == Phase.TRAIN:
            self.number_of_active_users = {(i, j): user for i, j, user in
                                           zip(range(0, blocks_num, conf.tasks_number),
                                               range(conf.tasks_number, blocks_num + conf.tasks_number,
                                                     conf.tasks_number),
                                               range(2, MAX_USERS + 1))}
        else:
            self.number_of_active_users = {(0, conf.test_blocks_num): N_USER}

    def get_current_users(self, index: int):
        for start, end in self.number_of_active_users.keys():
            if index in range(start, end):
                return self.number_of_active_users[(start, end)]
        return -1
