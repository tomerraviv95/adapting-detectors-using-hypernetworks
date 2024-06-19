from python_code.utils.constants import Phase, MAX_USERS, TRAINING_BLOCKS_PER_CONFIG

NUMBER_OF_USERS_DICT = {
    Phase.TRAIN: {(i, j): user for i, j, user in
                  zip(list(range(0, TRAINING_BLOCKS_PER_CONFIG * MAX_USERS, TRAINING_BLOCKS_PER_CONFIG)),
                      list(range(TRAINING_BLOCKS_PER_CONFIG, TRAINING_BLOCKS_PER_CONFIG * (MAX_USERS + 1),
                                 TRAINING_BLOCKS_PER_CONFIG)),
                      range(2, MAX_USERS + 1))},
    Phase.TEST: {(0, 20): 14, (20, 40): 14, (40, 60): 14, (60, 80): 14, (80, 100): 14}}


class UsersNetwork:
    def __init__(self, phase: Phase):
        self.number_of_active_users = NUMBER_OF_USERS_DICT[phase]

    def get_current_users(self, index: int):
        for start, end in self.number_of_active_users.keys():
            if index in range(start, end):
                return self.number_of_active_users[(start, end)]
        return -1
