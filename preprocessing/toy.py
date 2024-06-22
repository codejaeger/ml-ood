import numpy as np
import matplotlib.pyplot as plt
from utils.utils import project_and_filter
import sklearn

def train_data():
	r_quad = 80 + np.sin(np.linspace(0, np.pi, 100)) * 40

	all_samples = []
	Y = []
	for quads in range(6):
		thet_quad = np.linspace(0, np.pi/3, 100) + quads * np.pi/3
		mus = np.vstack([r_quad * np.sin(thet_quad), r_quad * np.cos(thet_quad)]).T
		samples = np.random.normal(size=(5*100*2)).reshape(5, 100, 2) * 13 * np.random.uniform(size=(5*100*2)).reshape(5, 100, 2) + \
					mus[None, :, :]
		all_samples.append(samples.reshape(-1, 2))

	all_samples = np.concatenate(all_samples, axis=0)

	Y += [1]*len(all_samples)

	r_quad = 40 + np.sin(np.linspace(0, np.pi, 100)) * 25


	all_samplesn = []
	segs = 4
	for quads in range(segs):
		thet_quad = np.linspace(0, np.pi/(segs/2), 100) + quads * np.pi/(segs/2) + np.pi/6
		mus = np.vstack([r_quad * np.sin(thet_quad), r_quad * np.cos(thet_quad)]).T
		samples = np.random.normal(size=(4*100*2)).reshape(4, 100, 2) * 9 * np.random.uniform(size=(4*100*2)).reshape(4, 100, 2) + \
					mus[None, :, :]
		all_samplesn.append(samples.reshape(-1, 2))

	all_samplesn = np.concatenate(all_samplesn, axis=0)
	Y += [0]*len(all_samplesn)

	return np.concatenate([all_samples, all_samplesn], axis=0), np.array(Y)

def val_data():
	r_quad = 65 + np.sin(np.linspace(0, np.pi, 100)) * 40

	all_samples = []
	t_Y = []
	for quads in range(6):
		thet_quad = np.linspace(0, np.pi/3, 100) + quads * np.pi/3
		mus = np.vstack([r_quad * np.sin(thet_quad), r_quad * np.cos(thet_quad)]).T
		samples = np.random.normal(size=(5*100*2)).reshape(5, 100, 2) * 13 * np.random.uniform(size=(5*100*2)).reshape(5, 100, 2) + \
					mus[None, :, :]
		all_samples.append(samples.reshape(-1, 2))

	all_samples = np.concatenate(all_samples, axis=0)

	t_Y += [1]*len(all_samples)

	r_quad = 55 + np.sin(np.linspace(0, np.pi, 100)) * 25


	all_samplesn = []
	segs = 4
	for quads in range(segs):
		thet_quad = np.linspace(0, np.pi/(segs/2), 100) + quads * np.pi/(segs/2) + np.pi/6
		mus = np.vstack([r_quad * np.sin(thet_quad), r_quad * np.cos(thet_quad)]).T
		samples = np.random.normal(size=(4*100*2)).reshape(4, 100, 2) * 9 * np.random.uniform(size=(4*100*2)).reshape(4, 100, 2) + \
					mus[None, :, :]
		all_samplesn.append(samples.reshape(-1, 2))

	all_samplesn = np.concatenate(all_samplesn, axis=0)
	t_Y += [0]*len(all_samplesn)
 
	return np.concatenate([all_samples, all_samplesn], axis=0), np.array(t_Y)

def test_data():
	tX, tY = val_data()
	ang = np.random.rand() * 2*np.pi
	tX = tX @ np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]).T

	return tX, tY

def plot_data(trX, trY, vX, vY, tX, tY, args):
	plt.figure()
	colors1 = np.array(['green', 'red'])
	colors2 = np.array(['yellow', 'maroon'])
	colors3 = np.array(['blue', 'violet'])
    
	plt.scatter(trX[:, 0], trX[:, 1], c=[colors1[1-_y] for _y in trY], s=0.1)
	plt.scatter(vX[:, 0], vX[:, 1], c=[colors2[1-_y] for _y in vY], s=0.1)
	plt.scatter(tX[:, 0], tX[:, 1], c=[colors3[1-_y] for _y in tY], s=0.1)
	
	x = [-150, -100, -50, 0, 50, 100, 150]
	y = [-100, -50, 0, 50, 100]
	# create an index for each tick position
	xi = list(range(len(x)))
	plt.xticks(x, x)
	plt.yticks(y, y)
	plt.title("Toy OOD data")
	plt.savefig(f"{args.outdir}/toy/toy-ood.png")

def plot_one(x, y, colors: list, title: str, args):
	plt.figure()
	plt.scatter(x[:, 0], x[:, 1], c=[colors[_y] for _y in y], s=0.1)
	
	x = [-150, -100, -50, 0, 50, 100, 150]
	y = [-100, -50, 0, 50, 100]
	# create an index for each tick position
	xi = list(range(len(x)))
	plt.xticks(x, x)
	plt.yticks(y, y)
	plt.savefig(f"{args.outdir}/toy/{title}.png")

def parse_data(args, config):
	import os
	os.makedirs(f'{args.outdir}/toy/', exist_ok=True)
	trX, trY = train_data()
	vX, vY = val_data()
	tX, tY = test_data()
	scaler = sklearn.preprocessing.StandardScaler(with_std=config.toy.use_std)
	scaler.fit(trX)
	trX = scaler.transform(trX)
	tX = scaler.transform(tX)
	vX = scaler.transform(vX)
	plot_data(trX, trY, vX, vY, tX, tY, args)
	
	X_sub, X_ids = project_and_filter(trX, np.array([1, 0]), 40)
	Y_sub = trY[X_ids]
	plot_one(X_sub, Y_sub, ['red', 'green'], "xproject", args)
	return trX, trY, vX, vY, tX, tY, scaler