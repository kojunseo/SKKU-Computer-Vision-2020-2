import time

class TimeModule:
    def __init__(self):
        self.now = time.time()

    def end_print(self, name):
        computational_time = time.time() - self.now
        print(f"Computational Time of {name} : {computational_time}")