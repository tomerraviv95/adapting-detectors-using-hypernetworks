from python_code.utils.constants import Phase, MAX_USERS, TRAINING_BLOCKS_PER_CONFIG

NUMBER_OF_USERS_DICT_TRAIN = {(i, j): user for i, j, user in
                              zip(range(0, TRAINING_BLOCKS_PER_CONFIG * MAX_USERS, TRAINING_BLOCKS_PER_CONFIG),
                                  range(TRAINING_BLOCKS_PER_CONFIG, TRAINING_BLOCKS_PER_CONFIG * (MAX_USERS + 1),
                                        TRAINING_BLOCKS_PER_CONFIG),
                                  range(2, MAX_USERS + 1))}

# Dict that describes the number of users in a sequence of blocks.
# Each key is (begin_block,end_block) with n_users value, such that users=n_users where begin_block <= t < end_block.
NUMBER_OF_USERS_DICT_TEST = {(0, 20): 14, (20, 40): 15, (40, 60): 16, (60, 80): 17, (80, 100): 18}


class UsersNetwork:
    def __init__(self, phase: Phase):
        self.number_of_active_users = NUMBER_OF_USERS_DICT_TRAIN if phase == Phase.TRAIN else NUMBER_OF_USERS_DICT_TEST

    def get_current_users(self, index: int):
        for start, end in self.number_of_active_users.keys():
            if index in range(start, end):
                return self.number_of_active_users[(start, end)]
        return -1
