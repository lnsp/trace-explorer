from abc import abstractmethod


class Transformer(object):
    @abstractmethod
    def columns(self) -> list[str]:
        pass

    @abstractmethod
    def transform(self, content: str) -> list:
        pass
