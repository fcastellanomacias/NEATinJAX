"""Every hyperparameter in the project lives here."""

# --------------------------------------------------------------- capacity ---
MAX_NODES = 40  # node slots per genome: inputs + bias + outputs + hidden
MAX_CONNS = 120  # connection slots per genome
W_CAP = 2.0  # connection weights are clipped to [-W_CAP, +W_CAP]

# ------------------------------------------------------------- population ---
POP_SIZE = 100
N_GENS = 3000
N_ELITES = 10  # top genomes copied unchanged into the next generation

# --------------------------------------------------------------- mutation ---
PROB_MUT_CONN = 0.8  # chance each connection weight gets perturbed
MUT_SIGMA = 0.3  # std dev of the Gaussian perturbation
PROB_ADD_CONN = 0.15  # chance an offspring gains a connection
PROB_ADD_NODE = 0.10  # chance an offspring gains a hidden node

# -------------------------------------------------------------- crossover ---
PROB_CROSSOVER = 0.75  # chance an offspring comes from two parents
PROB_INHERIT_B = 0.50  # chance a matching gene takes its weight from parent B

# ------------------------------------------------------------- speciation ---
C_UNMATCHED = 1.0  # weight on the unmatched-genes term of the distance
C_WEIGHT = 0.4  # weight on the mean-weight-difference term
COMPAT_THRESHOLD = 0.55  # distance below which two genomes share a species
CULL_RATIO = 0.5  # fraction of each species discarded before breeding

# --------------------------------------------------------------- pressure ---
COMPLEXITY_PENALTY = 0.05  # fitness cost per active connection
ANNEAL_PENALTY = True  # linearly decay that goes to zero over the run

# ------------------------------------------------- Slime Volleyball task ----
N_EPISODES = 3  # episodes averaged per genome per generation
MAX_STEPS = 1000  # steps per training episode
TEST_MAX_STEPS = 3000  # steps per evaluation match (ends at 5 lost lives)
SNAPSHOT_EVERY = 200  # stash the champion's topology this often
GIF_EVERY = 200  # render a GIF of the champion this often

# --------------------------------------------------- backprop NEAT task -----
BP_STEPS = 150  # gradient steps per genome per generation
BP_PENALTY = 0.01  # fitness cost on sqrt(number of active connections)
LEARN_RATE = 0.5  # gradient-descent step size
L2_REG = 1e-4  # L2 penalty on active weights
N_POINTS = 100  # samples per 2-D dataset
DATA_NOISE = 0.1  # Gaussian noise added to each dataset
DATASETS = ("gauss", "xor", "circle", "spiral")


# ------------------------------------------------------------------------ ---
# Presets. Values not listed keep the defaults above.
# ------------------------------------------------------------------------ ---
PRESETS = {
    # NEAT against the built-in Slime Volleyball AI
    "slimevolley": dict(
        MAX_NODES=40,
        MAX_CONNS=120,
        POP_SIZE=100,
        N_GENS=3000,
        N_ELITES=10,
        PROB_CROSSOVER=0.75,
        COMPLEXITY_PENALTY=0.05,
        ANNEAL_PENALTY=True,
        SNAPSHOT_EVERY=200,
        GIF_EVERY=200,
    ),
    # Backprop NEAT on 2-D classification
    "classification": dict(
        MAX_NODES=20,
        MAX_CONNS=60,
        POP_SIZE=30,
        N_GENS=40,
        N_ELITES=3,
        PROB_CROSSOVER=0.80,
        COMPLEXITY_PENALTY=0.0,
        ANNEAL_PENALTY=False,
        SNAPSHOT_EVERY=10,
    ),
}


def use(preset, **overrides):
    """Activate a preset, then apply any keyword overrides."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")

    settings = dict(PRESETS[preset])
    settings.update(overrides)

    g = globals()
    for name, value in settings.items():
        if (
            name not in g
            or name.startswith("_")
            or name in ("PRESETS", "use", "summary")
        ):
            raise ValueError(f"unknown hyperparameter {name!r}")
        g[name] = value

    if g["MAX_NODES"] < 1 or g["MAX_CONNS"] < 1:
        raise ValueError("MAX_NODES and MAX_CONNS must be positive")


def summary():
    """The active settings, as a printable multi-line string."""
    g = globals()
    names = [n for n in g if n.isupper() and not n.startswith("_") and n != "PRESETS"]
    width = max(len(n) for n in names)
    return "\n".join(f"  {n:<{width}} = {g[n]}" for n in sorted(names))
