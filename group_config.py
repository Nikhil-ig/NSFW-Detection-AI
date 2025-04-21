class GroupConfigManager:
    def __init__(self):
        self.group_settings = {}

    def toggle_enabled(self, group_id):
        self.group_settings[group_id] = not self.group_settings.get(group_id, False)
        return self.group_settings[group_id]

    def is_enabled(self, group_id):
        return self.group_settings.get(group_id, False)

    def auto_ban_enabled(self, group_id):
        return True  # Set to True by default; customize later
