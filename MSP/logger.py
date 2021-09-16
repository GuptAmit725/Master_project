import os

class logs:
    def __init__(self):
        pass
    def log(self, message):
        with open('static/logs.txt','a') as f:
            f.write(f"{message} \n")

    def copy_log(self):
        with open('static/logs.txt','r') as f:
            with open('static/logs_copy.txt', "w") as f1:
                for line in f:
                    f1.write(line)

    def erase_file(self):
        try:
            os.remove('static/logs.txt')
        except OSError as e:
            pass
