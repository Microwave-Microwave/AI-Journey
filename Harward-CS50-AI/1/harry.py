from logic import *

rain = Symbol("rain") #It is raining.
hagrid = Symbol("hagrid") #Harry visited Hagrid.
dumbledore = Symbol("dumbledore") #Harry visited Dumbledore.

knowledge = And(
    Implication(Not(rain), hagrid),
    Or(hagrid, dumbledore),
    Not(And(hagrid, dumbledore)),
    dumbledore
)

mustard = Symbol("mustard")
plum = Symbol("plum")
scarlet = Symbol("scarlet")

ballroom = Symbol("ballroom")
kitchen = Symbol("kitchen")
library = Symbol("library")

knife = Symbol("library")
revolver = Symbol("library")
wrench = Symbol("library")

knowledge2 = And(
    Or(mustard, Or(plum, scarlet)),
    Or(ballroom, Or(kitchen, library)),
    Or(knife, Or(revolver, wrench)),
    Not(plum),
    Or(Not(mustard), Or(Not(library), Not(revolver)))
)

print(model_check(knowledge, rain))