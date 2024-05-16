import time


class Announce:
    tab = 0

    def __init__(self):
        pass

    @staticmethod
    def print_time():
        return "[" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "]"

    @staticmethod
    def get_time():
        return time.strftime("%Y%m%d_%H%M%S", time.localtime())

    @staticmethod
    def printMessage():
        str = ''
        for i in range(0, Announce.tab):
            str += '  '
        return str + Announce.print_time()

    @staticmethod
    def space():
        str = ''
        for i in range(0, Announce.tab):
            str += '  '
        return str

    @staticmethod
    def doing():
        str = ''
        for i in range(0, Announce.tab):
            str += '  '
        Announce.tab += 1
        return str + Announce.print_time()

    @staticmethod
    def done():
        str = ''
        Announce.tab -= 1
        for i in range(0, Announce.tab):
            str += '  '
        return str + Announce.print_time() + ' done'

    def message(self, s, setting):
        # print(getTime(), s, sep='', file=setting.logFile)
        print(self.print_time(), s, sep='', file=setting.logFile)
        setting.logFile.flush()


class Progress:
    def __init__(self, step=5000):
        self.done = 0
        self.count = 0
        self.step = step

    def update(self, *args):
        self.done += 1
        self.count += 1
        if self.count >= self.step:
            self.count = 0
            print(Announce.print_time(), args, self.count, self.done)


if __name__ == '__main__':
    print(Announce.get_time())
