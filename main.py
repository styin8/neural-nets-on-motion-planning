import models

class Option():
    model = "base"
opt = Option()

m = models.create_model(opt=opt)
m.get()