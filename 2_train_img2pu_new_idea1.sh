python 2_train_img2pu_new_idea1.py \
	--data_dir_imgs data/shapenet/ShapeNetRendering \
	--data_dir_pcl data/shapenet/ShapeNet_pointclouds \
	--exp expts/2/idea1/img2pu_idea1_1to2_cabinet_cd \
	--gpu 0 \
	--ae_logs expts/2/gentest/pa_up2_1024_cabinet \
	--category cabinet \
	--bottleneck 512 \
	--up_ratio 2 \
	--loss chamfer \
	--batch_size 32 \
	--lr 5e-5 \
	--bn_decoder \
	--load_best_ae \
	--max_epoch 20 \
	--print_n 100
	# --sanity_check
