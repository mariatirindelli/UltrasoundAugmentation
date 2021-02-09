class BaseMethod():
    def __init__(self):
        self.name = ""

    def execute(self, image):
        raise NotImplementedError

