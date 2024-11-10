# status_manager.py
class StatusManager:
    _instance = None

    def __init__(self):
        self.status = 0

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StatusManager, cls).__new__(cls)
            cls._instance.status = 0  # 初始状态为 0
        return cls._instance

    def set_status(self, status_code: int):
        self.status = status_code

    def get_status(self):
        return self.status
