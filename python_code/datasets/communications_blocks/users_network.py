from python_code.utils.constants import Phase, MAX_USERS, TRAINING_SYMBOLS

NUMBER_OF_USERS_DICT = {Phase.TRAIN: {(i, j): user for i, j, user in zip(list(range(0, TRAINING_SYMBOLS * MAX_USERS, TRAINING_SYMBOLS)),
                                                                         list(range(TRAINING_SYMBOLS, TRAINING_SYMBOLS * (MAX_USERS + 1), TRAINING_SYMBOLS)),
                                                                         range(2, MAX_USERS + 1))},
                        Phase.TEST: {(0, 20): 6, (20, 40): 7, (40, 60): 8, (60, 80): 9, (80, 100): 10}}


class UsersNetwork:
    def __init__(self, phase: Phase):
        self.number_of_active_users = NUMBER_OF_USERS_DICT[phase]

    def get_current_users(self, index: int):
        for start, end in self.number_of_active_users.keys():
            if index in range(start, end):
                return self.number_of_active_users[(start, end)]
        return -1
