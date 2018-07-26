import datetime


class Stopwatch:
    def __init__(self, set_print=True):
        self.name = ""
        self.__set_print = set_print
        self.__start_time = None
        self.__split_list = []

    def turn_off_print(self):
        self.__set_print = False

    def turn_on_print(self):
        self.__set_print = True

    def start(self, name=""):
        self.name = name
        self.__start_time = self.__now()
        self.__print_to_screen("Started stopwatch %s" % name)

    def split(self, name=""):
        current_time = self.__now()
        self.__split_list.append(current_time)
        self.__print_to_screen("\tSplit %s" % ("(%s)" % name))
        self.__print_to_screen("\t\tTime: %s" % (current_time - self.__start_time))

    def reset(self):
        current_time = self.__now()
        self.__print_to_screen("Reset stopwatch %s at: %s" % (self.name,  (current_time - self.__start_time)))
        self.__reset()

    @staticmethod
    def __now():
        return datetime.datetime.now()

    def __reset(self):
        self.__start_time = None
        self.__split_list = []

    def __print_to_screen(self, text):
        if self.__set_print:
            print(text)


global stopwatch
stopwatch = Stopwatch()

if __name__ == "__main__":
    stopwatch.start()
    stopwatch.split("event a")
    stopwatch.split("event b")
    stopwatch.reset()
