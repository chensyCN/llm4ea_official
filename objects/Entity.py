class Entity:
    def __init__(self, idx: int, name: str, preprocess_func, is_literal=False, affiliation=None):
        self._is_literal = is_literal

        self.id: int = idx
        self.name: str = name.strip()
        self.value = None

        self.preprocess_func = preprocess_func
        self.affiliation = affiliation

        self.involved_as_tail_dict = dict()
        self.involved_as_head_dict = dict()

        self.embedding = None

        self.__init()

    @staticmethod
    def is_entity():
        return True

    @staticmethod
    def is_relation():
        return False

    def __init(self):
        self.value = self.preprocess_func(self.name)

    def is_literal(self):
        return self._is_literal

    def add_relation_as_head(self, relation, tail):
        if self.involved_as_head_dict.__contains__(relation) is False:
            self.involved_as_head_dict[relation] = set()
        self.involved_as_head_dict[relation].add(tail)

    def add_relation_as_tail(self, relation, head):
        if self.involved_as_tail_dict.__contains__(relation) is False:
            self.involved_as_tail_dict[relation] = set()
        self.involved_as_tail_dict[relation].add(head)

