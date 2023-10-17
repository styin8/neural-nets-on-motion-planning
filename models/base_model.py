class BaseModel():
    def __init__(self, opt) -> None:
        pass
    def get(self):
        print(self.__module__)
        return self.__module__