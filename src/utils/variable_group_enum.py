from enum import StrEnum
import re
class VariableGroup(StrEnum):
     LATENT = "z"
     OBSERVED = "x"

     def get_time_step(self, t: int):
         return f"{self.value}_{t}"

     @staticmethod
     def is_time_step(name: str,value: str):
         return re.match(f"{value}_\d+", name) is not None
     def is_value(self, name: str):
         return self.is_time_step(name, str(self.value))

     @classmethod
     def is_latent(cls, name: str):
         return cls.is_time_step(name, str(cls.LATENT))

     @classmethod
     def is_observed(cls, name: str):
            return cls.is_time_step(name, str(cls.OBSERVED))

     @classmethod
     def  is_variable_group(cls, name: str):
         return cls.is_latent(name) or cls.is_observed(name)




V = VariableGroup