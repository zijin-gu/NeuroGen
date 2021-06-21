import argparse
import time
import torch
from torch import nn
import numpy as np
import os
from visualize import center_crop, save_image, save_gif
from encoding import load_encoding
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_int, truncated_noise_sample)
import matplotlib.pyplot as plt

def get_args():
	
	# Init a parser.
	parser = argparse.ArgumentParser (
		prog='NeuroGen', 
		description='Provide an ROI ID to get the optimized images that maxmize its activation.'
	)
	
	# Add arguments to parser.
	parser.add_argument('--roi', type=int, default=0, help='ROI ID, range=[0, 24]') 
	parser.add_argument('--steps', type=int, default=1000, help='number of generations for the optimization.')
	parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
	parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
	parser.add_argument('--subj', type=int, default=1, help='subject ID, rangr [1,8]')
	parser.add_argument('--reptime', type=int, default=1, help='number of repeatations') 
	parser.add_argument('--truncation', type=int, default=1, help='truncation parameter')    
	args = parser.parse_args()

	return args

def neurogen(model, classifier, maps, num_class, roi, num_steps=1000, lr=0.01, wdecay=0.0001, dims=(227, 227, 3), device=torch.device("cpu"), repeat_time=1, truncation=1):
	
	# Init a random code to start from.
	code = truncated_noise_sample(batch_size=1, truncation=truncation, seed=repeat_time)
	code = torch.from_numpy(code).to(device) 
	class_vector =  one_hot_from_int([num_class], batch_size=1)
	class_vector = torch.from_numpy(class_vector).to(device) 


	optimizer = torch.optim.Adam([code.requires_grad_()], lr=lr)
	
	# Make sure we're in evaluation mode.
	model.eval()
	classifier.eval()
	maps.eval()

	step = 0
	keep_imgs = []
	step_loss = []
	keep_act = [] 
	keep_code = []    
	while step < num_steps:
		step += 1
		meta = 0

		def closure():

			optimizer.zero_grad()

			# Produce an image from the code and the conditional class.          
			y = model(code, class_vector, truncation)
			# Normalize said image s.t. values are between 0 and 1.
			y = (y + 1.0 ) / 2.0
			y = center_crop(y, 256, 227)
			# Try to classify this image
			pred = classifier(maps(y))

			out = pred[roi]
			
			# Get the loss with L2 weight decay.
			loss = -out + wdecay * torch.sum( code**2 )

			#loss.backward(retain_graph=True)
			loss.backward()
			
		   
			print("Step %d"%step, 
				"\n   loss  = {}".format(loss.data), 
				"\n   act   = {}".format(out.data), 
				"\n   code  = {}".format(code[0,:5].data))

			return loss
		
		optimizer.step(closure)
		step_loss.append(closure().data)

		y = model(code, class_vector, truncation)
		y = (y + 1.0 ) / 2.0
		y_crop = center_crop(y, 256, 227)
		act = classifier(maps(y_crop))
		keep_act.append(act[roi].cpu().detach().numpy())
		y = np.moveaxis(y.cpu().detach().numpy()[0], 0, -1)
		keep_imgs.append(y)
		keep_code.append(code.cpu().detach().numpy()[0])
		

	opt_step = np.argmax(keep_act)
	out_img = keep_imgs[opt_step]
	out_code = keep_code[opt_step]
	out_act = keep_act[opt_step]    
    
	print("Optimal step is ", opt_step)
	print("Optimal act  is ", out_act)
	return out_img, keep_imgs, keep_act, out_code, out_act

def main():
	# Pull some arguments from the CL.
	args = get_args()
	now = time.ctime()

	device = torch.device("cuda:%d"%args.gpu if torch.cuda.is_available() else "cpu")

	fwrf, fmaps = load_encoding(subject=args.subj, model_name='dnn_fwrf', device=device)

	model = BigGAN.from_pretrained('biggan-deep-256')
	model.to(device)

	top_idx = np.load('./img/S%d'%args.subj + '/top10_class.npy')
	top_idx = top_idx[:,args.roi]

	#start
	all_act = np.zeros([10,args.reptime])
	for cate in range(10):
		for repeat in range(args.reptime):
			begin = time.time()
			sim_image, sim_video, keep_act, final_code, final_act = neurogen(
				model=model, 
				classifier=fwrf,
				maps=fmaps, 
				num_class=top_idx[cate],
				roi=args._class,
				num_steps=args.steps,
				lr=args.lr, 
				wdecay=0.001,
				device=device, 
				repeat_time=repeat,
				truncation=args.truncation)
			end = time.time()

			# Let me know when things have finished processing.
			print('[INFO] Completed processing SIM in {:0.4}(s)!! Requested ROI {} '.format(end - begin, args.roi)) 

			# Save/Show the image.
			save_image(img=sim_image, subject=args.subj, category=top_idx[cate], repeat=repeat, roi=args.roi)
			#save_gif(img=sim_video, subject=args.subj,  repeat=repeat, roi=args.roi)

			output_dir = './img/S%0d'%args.subj + '/ROI%02d'%args.roi + '/'
			if not os.path.exists(output_dir):
				os.makedirs(output_dir)
			all_act[cate, repeat] = final_act
	np.save(output_dir + 'all_activations.npy', all_act)

if __name__ == "__main__":
	main()



