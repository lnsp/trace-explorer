from abc import abstractmethod

class Transformer(object):
    @abstractmethod
    def columns() -> list[str]:
        return ['scan', 'filter']


    @abstractmethod
    def transform(filename: str, line: str) -> list:
        pass