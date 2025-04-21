from collections import defaultdict

class UserManager:
    def __init__(self):
        self.warnings = defaultdict(lambda: defaultdict(int))

    def increment_warning(self, group_id, user_id):
        self.warnings[group_id][user_id] += 1
        return self.warnings[group_id][user_id]

    def get_warnings(self, group_id, user_id):
        return self.warnings[group_id][user_id]
