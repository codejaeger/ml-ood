import tensorflow as tf
import numpy as np

class RandFeats:
  def __init__(self, X, sigma_rot, bw, d, D=128):

    self.sigmas = [sigma_rot * _bw for _bw in bw]
    self.D = D
    self.Ws = []
    for sigma in self.sigmas:
      self.Ws.append(np.float32(np.random.randn(d, D)/sigma))
    self.Ws = np.stack(self.Ws, 0)

  def get_features(self, x_in):
    # phis = []
    # TODO: vectorize
    # for W in Ws:
    #   XW = np.matmul(x_in, W)
    #   phis.append(
    #     np.concatenate([np.sin(XW), np.cos(XW)], -1))
    # return np.concatenate(phis, -1)
    phis = tf.matmul(x_in, self.Ws)  # k x N x D
    phis = tf.transpose(phis, [1, 2, 0])  # N x D x k
    phis = tf.concat((tf.sin(phis), tf.cos(phis)), 1)
    return tf.reshape(phis, [x_in.shape[0], -1])

  def __call__(self, x_in):
    return self.get_features(x_in)

class RandFeatsTruncated:
  def __init__(self, X, sigma_rot, bw, d, D=128):

    self.sigmas = [sigma_rot * _bw for _bw in bw]
    self.D = D
    self.Ws = []
    for sigma in self.sigmas:
      self.Ws.append(np.float32(np.random.randn(d, D)/sigma))
    self.Ws = np.stack(self.Ws, 0)
    self.Ws = self.sample_features(X)
    
  def sample_features(self, X, ):
    L = int(0.3 * len(X)) # number of inputs to sample
    M = self.Ws.shape[0] * self.Ws.shape[2] 
    N = int(M/100)+10 # number of features to sample
    phi_Xt = tf.transpose(self.get_features(X[np.random.choice(len(X), L)])) / np.sqrt(L) # MxL
    phi_phi_T = phi_Xt @ tf.transpose(phi_Xt) # MxM
    mu = np.power(10, 0) # regularisation to use pow(10, -7) to pow(10, 1)
    diag = np.diag(phi_phi_T @ np.linalg.inv(phi_phi_T + mu))
    diag = diag / np.sum(diag)
    # print(M, len(diag)//2, phi_Xt.shape, self.Ws.shape)
    # print("Diag", M, diag, np.argsort(diag)[-N])
    _, Nd, _ = self.Ws.shape
    diag = diag[:len(diag)//2] + diag[len(diag)//2:]
    w_indices = np.unique(np.argsort(diag)[-N:])
    _Ws = np.reshape(np.transpose(self.Ws, [1, 2, 0]), (Nd, -1))[:, w_indices]
    return np.transpose(np.reshape(_Ws, (Nd, -1, 1)), [2, 0, 1])

  def get_features(self, x_in):
    # phis = []
    # TODO: vectorize
    # for W in Ws:
    #   XW = np.matmul(x_in, W)
    #   phis.append(
    #     np.concatenate([np.sin(XW), np.cos(XW)], -1))
    # return np.concatenate(phis, -1)
    phis = tf.matmul(x_in, self.Ws)  # k x N x D
    # phis = tf.transpose(phis, [1, 2, 0])  # N x D x k
    phis = tf.transpose(phis, [1, 2, 0])[:, None, :, :]  # N x 1 x D x k
    phis = tf.concat((tf.sin(phis), tf.cos(phis)), 1) # N x 2 x D x k
    return tf.reshape(phis, [x_in.shape[0], -1]) # Nx (D*k + D*k)

  def __call__(self, x_in):
    return self.get_features(x_in)

def define_rand_feats(X, xD, bw, args):
  """
  Args:
    ndata_feats: scalar value of total number of data features
    nrand_feats: scalar value of total number of desired random features
    gamma: Float, scale of frequencies

  Returns:
    Ws: ndata_feats x nrand_feats weight matrix
    bs: 1 x nrand_feats bias vector
  """
  tf.random.set_seed(123129) # For reproducibility
  from scipy.spatial import distance
  rprm = np.random.permutation(X.shape[0])
  ds = distance.cdist(X[rprm[:100], :], X[rprm[100:], :])
  sigma_rot = np.mean(np.sort(ds)[:, 5])
  if args.rff_algorithm == "vanilla":
    model = RandFeats(X, sigma_rot, bw, X.shape[1], int(X.shape[1]*xD))
  elif args.rff_algorithm == "truncated":
    model = RandFeatsTruncated(X, sigma_rot, bw, X.shape[1], int(X.shape[1]*xD))

  return model