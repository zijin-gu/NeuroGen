import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import imageio
import numpy as np
import os
def convert(img, target_type_min, target_type_max, target_type):
	imin = img.min()
	imax = img.max()

	a = (target_type_max - target_type_min) / (imax - imin)
	b = target_type_max - a * imax
	new_img = (a * img + b).astype(target_type)

	return new_img


def save_image(img, subject, category, repeat, roi):

	plt.figure()
	plt.imshow(img, aspect='equal')
	plt.tight_layout()
	plt.axis('off')
	output_dir = './img/S%0d'%subject + '/ROI%02d'%roi + '/'
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	plt.imsave(output_dir + 'C%04d'%category + '_repeat%d'%repeat + '.png', img, format='png')

	return


def save_gif(img, subject, category, repeat, roi):

	fig = plt.figure()
	plt.tight_layout()
	plt.axis('off')
	ima = []
	for cur in img:
		im = plt.imshow(cur, animated=True, aspect='equal')
		ima.append( [im] )
	ani = ArtistAnimation(fig, ima, interval=30, blit=True)
    
	output_dir = output_dir = './img/S%0d'%subject + '/ROI%02d'%roi + '/'
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	img_gif = convert(np.asarray(img), 0, 255, np.uint8)
	imageio.mimwrite(output_dir + 'C%04d'%category + '_repeat%d'%repeat + '.png', img_gif, fps=32)
		
	return
	

