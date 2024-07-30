from enum import StrEnum

class VariableGroup(StrEnum):
     LATENT = "z"
     OBSERVED = "x"

V = VariableGroup