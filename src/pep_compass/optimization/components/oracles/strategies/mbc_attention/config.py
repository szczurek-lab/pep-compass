# 14 chosen descriptors from the original paper.
def_fts = [
    "type8raac9glmd3lambda-correlation", "type8raac7glmd3lambda-correlation",
    "QSOrder_lmd4", "QSOrder_lmd3", "QSOrder_lmd2", "QSOrder_lmd1", "QSOrder_lmd0",
    "type5raac15glmd4lambda-correlation", "type7raac10glmd3lambda-correlation",
    "type5raac8glmd2lambda-correlation", "type3Braac9glmd3lambda-correlation",
    "type2raac15glmd4lambda-correlation", "type2raac8glmd2lambda-correlation",
    "type8raac14glmd1lambda-correlation"
]

# Scaling.
def_bias = 3.8
def_scale = 1/6

# Minimum sequence length required by QSOrder feature.
min_seq_length = 5
max_seq_length = 60

batch_size = 64
