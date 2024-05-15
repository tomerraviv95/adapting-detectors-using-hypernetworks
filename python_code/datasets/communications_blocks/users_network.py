from python_code.utils.constants import Phase

NUMBER_OF_USERS_DICT = {Phase.TRAIN: {(0, 200): 6, (200, 400): 7, (400, 600): 8, (600, 800): 9, (800, 1000): 10},
                        Phase.TEST: {(0, 20): 10, (20, 40): 9, (40, 60): 8, (60, 80): 6, (80, 100): 7}}


class UsersNetwork:
    def __init__(self, phase: Phase):
        self.number_of_active_users = NUMBER_OF_USERS_DICT[phase]

    def get_current_users(self, index: int):
        for start, end in self.number_of_active_users.keys():
            if index in range(start, end):
                return self.number_of_active_users[(start, end)]
        return -1
