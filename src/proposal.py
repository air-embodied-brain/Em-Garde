import json
from dataclasses import dataclass
from typing import List, Dict


# ---------------- Clause ----------------

@dataclass
class Clause:
    literals: List[str]

    @staticmethod
    def from_string(s: str) -> "Clause":
        # Split by AND, trim spaces
        parts = [x.strip() for x in s.split("AND")]
        return Clause(literals=parts)

    def evaluate(self, values: Dict[str, bool]) -> bool:
        # Clause is TRUE iff *all* literals evaluate to true
        return all(values.get(lit, False) for lit in self.literals)

    def get_literals(self) -> List[str]:
        return self.literals[:]


# --------------- Main Expression ----------------

class Proposal:
    """
    input: an LLM-generated json list, in he format:
    [
        {"positive":"clause1 AND clause2 ..."},
        {"negative":"clause3 AND clause4 ..."},
        ...
    ]
    where each clause is a string showing a visual clue

    Evaluation rule:
        result = (any positive clause is True) AND (all negative clauses are False)
    """

    def __init__(self, json_list: List[Dict[str, str]]):
        self.positives: List[Clause] = []
        self.negatives: List[Clause] = []

        for entry in json_list:
            if "positive" in entry:
                self.positives.append(Clause.from_string(entry["positive"]))
            elif "negative" in entry:
                self.negatives.append(Clause.from_string(entry["negative"]))
            else:
                raise ValueError(f"Invalid entry: {entry}")

    def get_literals(self) -> List[str]:
        lits = []
        for c in self.positives + self.negatives:
            lits.extend(c.get_literals())
        # remove duplicates but keep order
        return list(dict.fromkeys(lits))
    
    def get_positive_literals(self) -> List[str]:
        lits = []
        for c in self.positives:
            lits.extend(c.get_literals())
        # remove duplicates but keep order
        return list(dict.fromkeys(lits))

    def evaluate(self, values: Dict[str, bool]) -> bool:
        # Evaluate positives
        positive_values = [c.evaluate(values) for c in self.positives]

        # Evaluate negatives
        negative_values = [c.evaluate(values) for c in self.negatives]

        # return (any(positive_values)) and (not any(negative_values))
        return (any(positive_values))            # ignor negatives for now
    
    
if __name__ == "__main__":
    # Example usage
    json_input = '''
    [{"positive":"A coin is placed on a weighing scale AND green numbers are displayed on the scale"},
        {"positive":"Green numbers are displayed statically on a screen"},
        {"negative":"A person is holding a coin in his hand"},
        {"negative":"A person is closing the door of a weighing scale"},
        {"negative":"Green numbers change repidly on a screen"}]
    '''
    proposal = Proposal(json.loads(json_input))
    literals = proposal.get_literals()
    print("Literals:", literals)

    test_values = {
        "A coin is placed on a weighing scale": False,
        "green numbers are displayed on the scale": True,
        "Green numbers are displayed statically on a screen": True,
        "A person is holding a coin in his hand": False,
        "A person is closing the door of a weighing scale": False,
        "Green numbers change repidly on a screen": False
    }
    result = proposal.evaluate(test_values)
    print("Evaluation Result:", result)
    
    
