# conditional prior vae + adaptive algorithm (since random normal embedding for direction condition is difficult to learn
# , also the random normal space is sparse and high dimensional hence noisy)
## mean : f_theta(direction embedding linear layer)
## variance : f_phi(direction embedding linear layer)
## support loss : for a given direction embedding sample many latent vectors using mean and variance
# and minimise support loss

def get_learners(train_X, train_Y, bl, args, config):
    pass

def load_learners(args, config):
    pass