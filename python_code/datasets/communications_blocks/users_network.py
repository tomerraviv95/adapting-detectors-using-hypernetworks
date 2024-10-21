from python_code import conf
from python_code.utils.constants import Phase, MAX_USERS

N_USER = 14
NUMBER_OF_USERS_DICT_TRAIN = {(i, j): user for i, j, user in
                              zip(range(0, conf.tasks_number * MAX_USERS, conf.tasks_number),
                                  range(conf.tasks_number, conf.tasks_number * (MAX_USERS + 1),
                                        conf.tasks_number),
                                  range(2, MAX_USERS + 1))}

# Dict that describes the number of users in a sequence of blocks.
# Each key is (begin_block,end_block) with n_users value, such that users=n_users where begin_block <= t < end_block.
NUMBER_OF_USERS_DICT_TEST = {(0, 100): N_USER}


class UsersNetwork:
    def __init__(self, phase: Phase):
        self.number_of_active_users = NUMBER_OF_USERS_DICT_TRAIN if phase == Phase.TRAIN else NUMBER_OF_USERS_DICT_TEST

    def get_current_users(self, index: int):
        for start, end in self.number_of_active_users.keys():
            if index in range(start, end):
                return self.number_of_active_users[(start, end)]
        return -1
