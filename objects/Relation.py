
class Relation:
    def __init__(self, idx: int, name: str, preprocess_func, is_attribute=False, affiliation=None):
        self._is_attribute = is_attribute

        self.id: int = idx
        self.name: str = name.strip()
        self.value = None

        self.preprocess_func = preprocess_func
        self.affiliation = affiliation

        self.frequency = 0

        self.head_ent_set = set()
        self.tail_ent_set = set()
        self.tuple_set = set()

        self.functionality = 0.0
        self.functionality_inv = 0.0

        self.embedding = None
        self.__init()

    @staticmethod
    def is_entity():
        return False

    @staticmethod
    def is_relation():
        return True

    def __init(self):
        self.value = self.preprocess_func(self.name)

    def is_attribute(self):
        return self._is_attribute

    def add_relation_tuple(self, head, tail):
        self.head_ent_set.add(head)
        self.tail_ent_set.add(tail)
        self.tuple_set.add((head, tail))
        self.frequency += 1

    def calculate_functionality(self):
        if self.frequency == 0:
            return
        self.functionality = len(self.head_ent_set) / self.frequency
        self.functionality_inv = len(self.tail_ent_set) / self.frequency

