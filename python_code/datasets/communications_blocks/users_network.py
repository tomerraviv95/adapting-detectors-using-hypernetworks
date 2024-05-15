from python_code.utils.constants import Phase

NUMBER_OF_USERS_DICT = {Phase.TRAIN: {(0, 200): 4, (200, 400): 4, (400, 600): 4, (600, 800): 4, (800, 1000): 4},
                        Phase.TEST: {(0, 20): 4, (20, 60): 4, (60, 80): 4, (80, 100): 4}}


class UsersNetwork:
    def __init__(self, phase: Phase):
        self.number_of_active_users = NUMBER_OF_USERS_DICT[phase]

    def get_current_users(self, index: int):
        for start, end in self.number_of_active_users.keys():
            if index in range(start, end):
                return self.number_of_active_users[(start, end)]
        return -1
