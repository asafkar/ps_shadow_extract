"""
This will train a NN which will receive an input of photometric stereo + lights, and output a normal map

CUDA_VISIBLE_DEVICES=1,3 OMP_NUM_THREADS=4 python -m torch.distributed.run --nproc_per_node=2 train.py --use_clearml=True --base_dir=/home/akarnieli/fdata

"""
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from model import ViT_normals, ShadowDecoder, LightEst
import torchvision
import os, sys
import torch
import torch.distributed as dist
import argparse
from clearml import Task, Logger

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('pdf')

from torch import nn
import torch.utils.data
from datasets.blobby_dataset import Synth_Dataset
from datasets.blender_dataset import BlenderDataset
import logging
logging.getLogger('PIL').setLevel(logging.WARNING)


def eval_norm(model, args, loader, epoch, num_imgs=32, shadow_model=None, light_model=None, return_save=False):
	local_rank = int(os.environ['LOCAL_RANK'])
	
	model.eval()
	if shadow_model is not None:
		shadow_model.eval()
	if light_model is not None:
		light_model.eval()

	with torch.no_grad():
		cos_s = nn.CosineSimilarity(dim=1)
		running_acc = 0.0
		running_light_acc = 0.0
		running_shadow_acc = 0.0
		cnt = 0
		for iteration, sample in enumerate(loader):

			# idx = iteration
			norm_gt, imgs, silhouette = sample['normal_gt'], sample['imgs'], sample['silhouette']
			light = sample['light']
			norm_gt, imgs, silhouette, light = norm_gt.to(local_rank), imgs.to(local_rank), silhouette.to(local_rank), light.to(local_rank)

			if shadow_model is not None:
				shadows_gt = sample['shadows'].to(local_rank)

			# split to patches, feed to model, then combine back
			normal_hat, merged_features = model(imgs)
			normal_hat = torch.nn.functional.normalize(normal_hat, dim=1)

			if light_model is not None:
				light_hat = light_model(merged_features)
			if shadow_model is not None:
				shadows_hat = shadow_model(merged_features)

			normals = (normal_hat * silhouette).clamp(-1, 1)
			norm_gt *= silhouette
			cos_theta = cos_s(normals, norm_gt)

			ones = torch.ones_like(cos_theta)
			cos_theta = torch.where(cos_theta == 0, ones, cos_theta)  # cosine_sim returns 0 when compares 2 zero vectors
			error_map = torch.acos(cos_theta.clamp(-1, 1))  # [0, 2*pi]

			if shadow_model is not None:
				shadow_err = torch.abs(shadows_hat - shadows_gt).mean()
				running_shadow_acc += shadow_err

			else:
				light_err = torch.arccos(torch.dot(light_hat.flatten(), light.flatten())
					/ (torch.linalg.norm(light_hat.view(-1, 3)) * torch.linalg.norm(light.view(-1, 3))))
				running_light_acc += light_err

			normals = (normals + 1) / 2
			norm_gt = (norm_gt + 1) / 2

			angular_map = (error_map * 180.0 / np.pi) * silhouette
			mean_angle_err = angular_map.sum() / silhouette.sum()
			running_acc += mean_angle_err
			cnt += 1

		# draw some samples...
		# writer.add_image(f"eval normals_#{num_imgs}", normals[0], epoch)
		# writer.add_image(f"GT normals_#{num_imgs}", norm_gt[0], epoch)
		# writer.add_image(f"normal error map_#{num_imgs}", error_map[0].unsqueeze(0), epoch)
		idx_smp = np.random.randint(0, max(norm_gt.shape[0], 0))
		g = [norm_gt[idx_smp], normals[idx_smp], error_map[idx_smp].repeat(3, 1, 1)]
		grid1 = torchvision.utils.make_grid(g)
		writer.add_image(f"eval #_{num_imgs}: GT normal, pred_normal, err_map", grid1, epoch)

		if shadow_model is not None:
			bb, ss, _, _ = shadows_gt.shape
			bb = torch.randint(0, bb, [1])[0]
			ss = torch.randint(0, ss, [1])[0]
			g = [shadows_gt[bb, ss].unsqueeze(0), shadows_hat[bb, ss].unsqueeze(0), torch.abs(shadows_gt[bb, ss] - shadows_hat[bb, ss]).unsqueeze(0)]
			grid1 = torchvision.utils.make_grid(g)
			writer.add_image(f"eval #_{num_imgs}: GT shadow, pred_shadow, diff", grid1, epoch)
			running_shadow_acc /= cnt
			writer.add_scalar(f"eval: mean shadow error #{num_imgs}", running_shadow_acc, epoch)

		else:  # on none-blender data, not drawing shadows, so testing light.
			fig1 = plt.figure(1)
			light_hat_draw = light_hat.reshape(-1, 3).cpu()
			light_draw = light.reshape_as(light_hat_draw).cpu()
			plt.scatter(light_hat_draw[:, 0], light_hat_draw[:, 1], c=light_hat_draw[:, 2])
			plt.colorbar()
			for i, t in enumerate(light_hat_draw[:, 2]):
				plt.annotate(i, (light_hat_draw[:, 0][i], light_hat_draw[:, 1][i]))

			if args.use_clearml and local_rank==0:
				Logger.current_logger().report_matplotlib_figure(title="Images",
					series="Light Pred projections", iteration=epoch, figure=plt, report_image=True)
				fig1.clear()

				fig2 = plt.figure(1)
				plt.scatter(light_draw[:, 0], light_draw[:, 1], c=light_draw[:, 2])
				plt.colorbar()
				for i, t in enumerate(light_draw[:, 2]):
					plt.annotate(i, (light_draw[:, 0][i], light_draw[:, 1][i]))
				Logger.current_logger().report_matplotlib_figure(title="Images",
					series="Light GT projections", iteration=epoch, figure=plt, report_image=True)
				fig2.clear()

			running_light_acc /= cnt
			writer.add_scalar(f"eval: mean angle light error #{num_imgs}", running_light_acc, epoch)

		running_acc /= cnt
		writer.add_scalar(f"eval: mean angle error_#{num_imgs}", running_acc, epoch)
		print(f'validation set - mean angle error_#{num_imgs} = {running_acc}')

		if running_acc.item() < args.best_err and return_save:
			args.best_err = running_acc.item()
			return True
		return False


def get_data(args, local_rank, world_size):
	train_set = BlenderDataset(args, args.blender_data_dir, 'train', patch_size=128, num_val=5)
	train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=train_set, shuffle=False, rank=local_rank,num_replicas=world_size)
	blender_train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch, num_workers=args.workers, sampler=train_sampler,
			pin_memory=False, shuffle=False)

	val_set = BlenderDataset(args, args.blender_data_dir, 'test', num_val=5, patch_size=None)
	blender_test_loader = torch.utils.data.DataLoader(val_set, batch_size=args.val_batch, num_workers=args.workers, 
			pin_memory=False, shuffle=False)

	train_sets = []
	val_sets = []
	train_sets.append(Synth_Dataset(args, args.sculpture_data_dir, 'train', patch_size=64, subset=None))
	val_sets.append(Synth_Dataset(args, args.sculpture_data_dir, 'val', patch_size=None, subset=4))
	train_sets.append(Synth_Dataset(args, args.blobby_data_dir, 'train', patch_size=64, subset=None))
	val_sets.append(Synth_Dataset(args, args.blobby_data_dir, 'val', patch_size=None, subset=4))

	train_set_ = torch.utils.data.ConcatDataset(train_sets)
	val_set_ = torch.utils.data.ConcatDataset(val_sets)
	train_sampler_ = torch.utils.data.distributed.DistributedSampler(dataset=train_set_, shuffle=False, rank=local_rank,num_replicas=world_size)

	blobby_sculpt_train_loader = torch.utils.data.DataLoader(train_set_, batch_size=int(args.batch) * 4, sampler=train_sampler_,
			num_workers=args.workers, pin_memory=False, shuffle=False)
	blobby_sculpt_test_loader = torch.utils.data.DataLoader(val_set_, batch_size=args.val_batch, num_workers=args.workers, 
			pin_memory=False, shuffle=False)

	return blender_train_loader, blender_test_loader, blobby_sculpt_train_loader, blobby_sculpt_test_loader


def train_encoder_net(args):
	local_rank = int(os.environ['LOCAL_RANK'])
	world_size = int(os.environ['WORLD_SIZE'])
	if args.use_clearml and local_rank==0:
		task = Task.init(project_name="shadow_light_extraction_model", task_name=task_name)

	blender_train_loader, blender_test_loader, blobby_sculpt_train_loader, blobby_sculpt_test_loader = get_data(args, local_rank, world_size)

	dim = 3 * args.patch_size ** 2
	encoder_net = ViT_normals(
		args=args,
		image_size=128,
		patch_size=args.patch_size,
		dim=dim,
		depth=5,
		heads=16,
		mlp_dim=1024,
		dropout=0.1,
		emb_dropout=0.1,
		pool='cls'
	)
	encoder_net = encoder_net.to(local_rank)
	encoder_net = nn.parallel.DistributedDataParallel(encoder_net, device_ids=[local_rank], find_unused_parameters=False)

	shadow_net = ShadowDecoder(dim=dim, patch_size=args.patch_size, args=args).to(local_rank)
	shadow_net = nn.parallel.DistributedDataParallel(shadow_net, device_ids=[local_rank], find_unused_parameters=False)
	
	light_net = LightEst(dim=dim, args=args).to(local_rank)
	light_net = nn.parallel.DistributedDataParallel(light_net, device_ids=[local_rank], find_unused_parameters=False)

	# model_parameters = filter(lambda p: p.requires_grad, encoder_net.parameters())
	# params = sum([np.prod(p.size()) for p in model_parameters])
	# print(f"num of model parameters = {params}")

	cosine_loss = nn.CosineEmbeddingLoss()
	models_params = list(encoder_net.parameters()) + list(shadow_net.parameters()) + list(light_net.parameters())
	normal_optimizer = torch.optim.Adam(models_params, args.learning_rate, weight_decay=1e-7)

	normal_scheduler = torch.optim.lr_scheduler.StepLR(normal_optimizer, step_size=100, gamma=0.9, verbose=False)

	if args.checkpoint:
		print(f"Loading from checkpoint {args.checkpoint}")
		checkpoint = torch.load(args.checkpoint)
		encoder_net.load_state_dict(checkpoint['model_state_dict'])
		# normal_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		if not args.restart_epochs:
			args.start_epoch = checkpoint['epoch']
		
		shadow_net.load_state_dict(checkpoint['shadow_model_state_dict'])
		light_net.load_state_dict(checkpoint['light_model_state_dict'])

	for epoch in range(args.start_epoch-1, args.epochs+1):
		# blender_train_loader.sampler.set_epoch(epoch)
		print("Starting epoch {}".format(epoch))

		if epoch % 5 == 0 and local_rank == 0:
			# blender_test_loader.sampler.set_epoch(epoch)
			save_model = eval_norm(encoder_net, args, blender_test_loader, epoch, num_imgs=32, shadow_model=shadow_net,
					return_save=True)

			# eval_norm(encoder_net, args, blender_test_loader, epoch, num_imgs=24, shadow_model=shadow_net)
			# eval_norm(encoder_net, args, blender_test_loader, epoch, num_imgs=16, shadow_model=shadow_net)
			eval_norm(encoder_net, args, blobby_sculpt_test_loader, epoch, num_imgs=30, light_model=light_net)

			if save_model:
				print(f"saving model to {os.path.join(args.save_dir, args.task_name + '_snapshot.pth')}")
				torch.save({
					'epoch': epoch,
					'model_state_dict': encoder_net.state_dict(),
					'optimizer_state_dict': normal_optimizer.state_dict(),
					'shadow_model_state_dict': shadow_net.state_dict(),
					'light_model_state_dict': light_net.state_dict(),
				},
					os.path.join(args.save_dir, args.task_name + "_snapshot.pth"))

		encoder_net.train()  # Set model to training mode
		
		shadow_net.train()

		running_normal_loss = 0.0
		for iteration, sample in enumerate(blender_train_loader):
			# train_set.in_img_num = random.randint(24, 32)  # FIXME!!!!
			normal_gt, imgs, silhouette = sample['normal_gt'], sample['imgs'], sample['silhouette']
			normal_gt, imgs, silhouette = normal_gt.to(local_rank), imgs.to(local_rank), silhouette.to(local_rank)

			shadows_gt = sample['shadows'].to(local_rank)

			"""" Train normal net """
			encoder_net.zero_grad(set_to_none=True)
			shadow_net.zero_grad(set_to_none=True)
			light_net.zero_grad(set_to_none=True)

			normal_hat, merged_features = encoder_net(imgs)
			normal_hat_normed = torch.nn.functional.normalize(normal_hat, dim=1)

			# cosine loss
			encoder_net_loss = cosine_loss(normal_hat_normed.permute(0,2,3,1).contiguous().view(-1, 3),
							normal_gt.permute(0,2,3,1).contiguous().view(-1, 3),
							torch.ones(normal_hat.nelement()//3).to(local_rank))

			# light_err = torch.arccos(torch.dot(light_hat.flatten(), light.flatten())
			# 				/ (torch.linalg.norm(light_hat.view(-1, 3)) * torch.linalg.norm(light.view(-1, 3))))

			loss = encoder_net_loss
			# if not torch.isnan(light_err):
			# 	loss += 0.1 * light_err

			shadows = shadow_net(merged_features)
			loss += 5 * (torch.abs(shadows - shadows_gt).mean() + (torch.abs(shadows-shadows_gt)**2).mean())

			loss.backward()
			running_normal_loss += encoder_net_loss.item()
			normal_optimizer.step()

			if iteration == 50:  # print every 2000 mini-batches
				print('[%d, %5d] normals loss: %.5f' % (epoch, iteration + 1, running_normal_loss / iteration))
				writer.add_scalar("train: loss", running_normal_loss / iteration, epoch)
				running_normal_loss = 0.0

		normal_scheduler.step()
		writer.add_scalar("learning_rate", normal_scheduler.get_last_lr()[0], epoch)

		if epoch % 5 == 0:
			# blobby_sculpt_train_loader.sampler.set_epoch(epoch)
			for iteration, sample in enumerate(blobby_sculpt_train_loader):
				# train_set.in_img_num = random.randint(24, 32)  ### FIXME!!! 
				normal_gt, imgs, silhouette = sample['normal_gt'], sample['imgs'], sample['silhouette']
				normal_gt, imgs, silhouette = normal_gt.to(local_rank), imgs.to(local_rank), silhouette.to(local_rank)
				light = sample['light'].to(local_rank)

				"""" Train normal net """
				encoder_net.zero_grad(set_to_none=True)
				#if args.output_shadows:
				shadow_net.zero_grad(set_to_none=True)
				light_net.zero_grad(set_to_none=True)

				normal_hat, merged_features = encoder_net(imgs)
				light_hat = light_net(merged_features)
				normal_hat_normed = torch.nn.functional.normalize(normal_hat, dim=1)

				# cosine loss
				loss = cosine_loss(normal_hat_normed.permute(0, 2, 3, 1).contiguous().view(-1, 3),
					normal_gt.permute(0, 2, 3, 1).contiguous().view(-1, 3),
					torch.ones(normal_hat.nelement() // 3).to(local_rank))

				light_err = torch.arccos(torch.dot(light_hat.flatten(), light.flatten())
				                         / (torch.linalg.norm(light_hat.view(-1, 3)) * torch.linalg.norm(light.view(-1, 3)) + 1e-8)
				)

				if not torch.isnan(light_err):
					loss += 0.1 * light_err
				else:
					print(f"light_hat={light_hat}")
					print("!!! light err has nan")
					quit()

				loss.backward()
				normal_optimizer.step()


if __name__ == '__main__':
	def init_distributed(args):

		# Initializes the distributed backend which will take care of synchronizing nodes/GPUs
		env_dict = {
			key: os.environ[key]
			for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
		}
		print(f"[{os.getpid()}] Initializing process group with: {env_dict}")

		# # only works with torch.distributed.launch // torch.run
		rank = int(os.environ["RANK"])
		world_size = int(os.environ['WORLD_SIZE'])
		dist.init_process_group(
			backend="nccl",

			# init_method=dist_url,
			world_size=world_size,
			rank=rank
		)

		train_encoder_net(args)

		# Tear down the process group
		dist.destroy_process_group()

	print("Start training")
	print(f"Command ran: {str(sys.argv)}")
	parser = argparse.ArgumentParser(description='Description of your program')
	parser.add_argument('-lr', '--learning_rate', help='Description ', default=8e-5)
	parser.add_argument('-d', '--dev', help='device: cuda', required=False, default='cuda')
	parser.add_argument('--epochs', help='epochs', required=False, type=int, default=500)
	parser.add_argument('--start_epoch', default=1, type=int)
	parser.add_argument('--checkpoint', help='Checkpoint to continue training from', default=None)
	parser.add_argument('--save_dir', help='dir to save models ', default='/tmp/deep_shadow/models/')
	parser.add_argument('--use_clearml', help='Use clear-ml for logging ', default=False)
	parser.add_argument("--base_dir",  required=False, default='.')
	# parser.add_argument("--local_world_size", type=int, default=1)

	args = parser.parse_args()
	args.seed = 123
	args.best_err = 5000

	args.epochs = 1000
	args.batch = 10
	args.val_batch = 2
	args.workers = 8
	args.patch_size = 8
	args.rand_layer_skip = False
	args.restart_epochs = True  # when loading checkpoint, start from epoch 0
	args.dim = 16

	args.data_dir = f'{args.base_dir}/PS_Sculpture_Dataset/'
	args.save_dir = f'{args.base_dir}/models/'
	args.blender_data_dir = f'{args.base_dir}/blender_data/'
	args.blobby_data_dir = f'{args.base_dir}/PS_Blobby_Dataset/'
	args.sculpture_data_dir = f'{args.base_dir}/PS_Sculpture_Dataset/'

	task_name = f"shadow_normals_lights_depth5_heads16_patch_size8_batch10_lr{args.learning_rate}_shadow_conv_head"

	writer = SummaryWriter(task_name.replace(" ", "_"))

	torch.manual_seed(args.seed)
	args.task_name = task_name.replace(" ", "_")

	init_distributed(args)
