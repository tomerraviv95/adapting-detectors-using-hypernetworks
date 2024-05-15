from python_code.utils.constants import Phase

NUMBER_OF_USERS_DICT = {Phase.TRAIN: {(0, 200): 12, (200, 400): 13, (400, 600): 14, (600, 800): 15, (800, 1000): 16},
                        Phase.TEST: {(0, 20): 12, (20, 40): 13, (40, 60): 14, (60, 80): 15, (80, 100): 16}}


class UsersNetwork:
    def __init__(self, phase: Phase):
        self.number_of_active_users = NUMBER_OF_USERS_DICT[phase]

    def get_current_users(self, index: int):
        for start, end in self.number_of_active_users.keys():
            if index in range(start, end):
                return self.number_of_active_users[(start, end)]
        return -1
