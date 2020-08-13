from importer import *
from utils.encoder_2 import *
parser = argparse.ArgumentParser()

parser.add_argument('--data_dir_imgs', type=str, required=True, 
	help='Path to shapenet rendered images')
parser.add_argument('--data_dir_pcl', type=str, required=True, 
	help='Path to shapenet pointclouds')
parser.add_argument('--exp', type=str, required=True, 
	help='Name of Experiment')
parser.add_argument('--gpu', type=str, required=True, 
	help='GPU to use')
parser.add_argument('--ae_logs', type=str, required=True, 
	help='Location of pretrained auto-encoder snapshot')
parser.add_argument('--category', type=str, required=True, 
	help='Category to train on : \
		["all", "airplane", "bench", "cabinet", "car", "chair", "lamp", \
		"monitor", "rifle", "sofa", "speaker", "table", "telephone", "vessel"]')
parser.add_argument('--bottleneck', type=int, required=True, default=128, 
	help='latent space size')
parser.add_argument('--up_ratio', type=int, default=2, 
	help='up sampling ratio')
parser.add_argument('--loss', type=str, required=True, 
	help='Loss to optimize on l1/l2/chamfer')
parser.add_argument('--batch_size', type=int, default=32, 
	help='Batch Size during training')
parser.add_argument('--lr', type=float, default=0.00005, 
	help='Learning Rate')
parser.add_argument('--bn_decoder', action='store_true', 
	help='Supply this parameter if you want bn_decoder, otherwise ignore')
parser.add_argument('--load_best_ae', action='store_true', 
	help='supply this parameter to load best model from the auto-encoder')
parser.add_argument('--max_epoch', type=int, default=30, 
	help='max num of epoch')
parser.add_argument('--print_n', type=int, default=100, 
	help='print_n')
parser.add_argument('--sanity_check', action='store_true', 
	help='supply this parameter to visualize autoencoder reconstructions')

FLAGS = parser.parse_args()

print '-='*50
print FLAGS
print '-='*50

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

BATCH_SIZE = FLAGS.batch_size   	# Training Batch Size
VAL_BATCH_SIZE = FLAGS.batch_size   # Validation Batch Size
NUM_POINTS = 1024
NUM_EVAL_POINTS = 1024				# Number of points predicted
HEIGHT = 128 						# Height of input RGB image
WIDTH = 128 						# Width of input RGB image
GT_PCL_SIZE = 2048
UP_RATIO = FLAGS.up_ratio

def fetch_batch(models, indices, batch_num, batch_size):
	'''
	Input:
		models: list of paths to shapenet models
		indices: list of ind pairs, where 
			ind[0] : model index (range--> [0, len(models)-1])
			ind[1] : view index (range--> [0, NUM_VIEWS-1])
		batch_num: batch_num during epoch
		batch_size: batch size for training or validation
	Returns:
		batch_ip: input RGB image of shape (B, HEIGHT, WIDTH, 3)
		batch_gt: gt point cloud of shape (B, NUM_POINTS, 3)
	Description:
		Batch Loader
	'''

	batch_ip = []
	batch_gt = []

	for ind in indices[batch_num*batch_size:batch_num*batch_size+batch_size]:
		model_path = models[ind[0]]
		img_path = join(FLAGS.data_dir_imgs, model_path, 'rendering', PNG_FILES[ind[1]])
		pcl_path = join(FLAGS.data_dir_pcl, model_path, 'pointcloud_1024.npy')

		pcl_gt = np.load(pcl_path)

		ip_image = cv2.imread(img_path)[4:-5, 4:-5, :3]
		ip_image = cv2.cvtColor(ip_image, cv2.COLOR_BGR2RGB)

		batch_gt.append(pcl_gt)
		batch_ip.append(ip_image)

	return np.array(batch_ip), np.array(batch_gt)

def fetch_batch_dense(models, indices, batch_num, batch_size):
	'''
	Input:
		models: list of paths to shapenet models
		indices: list of ind pairs, where 
			ind[0] : model index (range--> [0, len(models)-1])
			ind[1] : view index (range--> [0, NUM_VIEWS-1])
		batch_num: batch_num during epoch
		batch_size: batch size for training or validation
	Returns:
		batch_ip: input RGB image of shape (B, HEIGHT, WIDTH, 3)
		batch_gt: gt point cloud of shape (B, NUM_POINTS, 3)
	Description:
		Batch Loader
	'''

	batch_ip = []
	#batch_gt = []
	batch_gt_den = []

	for ind in indices[batch_num*batch_size:batch_num*batch_size+batch_size]:
		model_path = models[ind[0]]
		img_path = join(FLAGS.data_dir_imgs, model_path, 'rendering', PNG_FILES[ind[1]])
		#pcl_path = join(FLAGS.data_dir_pcl, model_path, 'pointcloud_1024.npy')
		pcl_den_path = join(FLAGS.data_dir_pcl, model_path, 'pointcloud_2048.npy')

		#pcl_gt = np.load(pcl_path)
		pcl_gt_den = np.load(pcl_den_path)

		ip_image = cv2.imread(img_path)[4:-5, 4:-5, :3]
		ip_image = cv2.cvtColor(ip_image, cv2.COLOR_BGR2RGB)

		#batch_gt.append(pcl_gt)
		batch_ip.append(ip_image)
		batch_gt_den.append(pcl_gt_den)

	return np.array(batch_ip), np.array(batch_gt_den)

def get_epoch_loss(val_models, val_pair_indices):
	'''
	Input:
		val_models:	list of absolute paths to models in validation set
		val_pair_indices: list of ind pairs for validation set
		-->	ind[0] : model index (range--> [0, len(models)-1])
		-->	ind[1] : view index (range--> [0, NUM_VIEWS-1])
	Returns:
		val_chamfer: chamfer distance calculated on scaled prediction and gt
		val_forward: forward distance calculated on scaled prediction and gt
		val_backward: backward distance calculated on scaled prediction and gt
	Description:
		Calculate Val epoch metrics (chamfer, forward, backward, l1, l2)
		and log them to tensorboard
	'''

	batches = len(val_pair_indices)/VAL_BATCH_SIZE
	val_stats = {}
	val_stats = reset_stats(ph_summary, val_stats)

	for b in xrange(batches):
		batch_gt, batch_gt_den = fetch_batch_dense(val_models, val_pair_indices, b, VAL_BATCH_SIZE)
		runlist = [loss, chamfer_distance_rimg_scaled, dists_forward_rimg_scaled, dists_backward_rimg_scaled, emd]
		_l, C, F, B, E = sess.run(runlist, feed_dict={pcl_gt:batch_gt_den, img_inp:batch_ip})
		_summary_losses = [F, B, C, E, _l]

		val_stats = update_stats(ph_summary, _summary_losses, val_stats, batches)

	summ = sess.run(merged_summ, feed_dict=val_stats)
	return val_stats[ph_dists_chamfer], val_stats[ph_dists_forward], val_stats[ph_dists_backward], val_stats[ph_emd], summ


if __name__ == '__main__':

	# Create a folder for experiments and copy the training file
	create_folder(FLAGS.exp)
	train_filename = basename(__file__)
	os.system('cp %s %s'%(train_filename, FLAGS.exp))
	with open(join(FLAGS.exp, 'settings.txt'), 'w') as f:
		f.write(str(FLAGS)+'\n')

	# Create Placeholders
	img_inp = tf.placeholder(tf.float32, shape=(BATCH_SIZE, HEIGHT, WIDTH, 3), name='img_inp')
	#pcl_in = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINTS, 3), name='pcl_gt')
	pcl_gt = tf.placeholder(tf.float32, shape=(BATCH_SIZE, GT_PCL_SIZE, 3))
	is_training = True
	bn_decay = 0.95
	# Generate Prediction
	with tf.variable_scope('psgn') as scope:
		z_latent_img = image_encoder_se_pure(img_inp, FLAGS)

		out_img = decoder_with_fc_only(z_latent_img, layer_sizes=[512,1024,np.prod([NUM_POINTS, 3])],
			b_norm=FLAGS.bn_decoder,
			b_norm_finish=False,
			verbose=True,
			scope=scope
			)
	reconstr = tf.reshape(out_img, (BATCH_SIZE, NUM_POINTS, 3))

	with tf.variable_scope('generator') as scope:
		out, _ = get_gen_model(reconstr, is_training, scope=scope,
						reuse=None, use_normal=False, use_bn=False, use_ibn=False,
						bn_decay=bn_decay,up_ratio=UP_RATIO)
	# bneck_size = FLAGS.bottleneck
	# with tf.variable_scope('pointnet_ae') as scope:
	# 	z = encoder_with_convs_and_symmetry(in_signal=reconstr, n_filters=[64,128,128,256,bneck_size], 
	# 		filter_sizes=[1],
	# 		strides=[1],
	# 		b_norm=True,
	# 		verbose=True,
	# 		scope=scope
	# 		)
	# 	out = decoder_with_fc_only(z, layer_sizes=[512,1024,np.prod([NUM_POINTS, 3])],
	# 		b_norm=FLAGS.bn_decoder,
	# 		b_norm_finish=False,
	# 		verbose=True,
	# 		scope=scope
	# 		)

		# Point cloud reconstructed from input RGB image using latent matching network and fixed decoder from AE 
	reconstr_img = tf.reshape(out, (BATCH_SIZE, GT_PCL_SIZE, 3))

	# Calculate Chamfer Metrics reconstr_img <-> pcl_gt
	dists_forward_rimg, dists_backward_rimg, chamfer_distance_rimg = [tf.reduce_mean(metric) for metric in get_chamfer_metrics(pcl_gt, reconstr_img)]

	# Calculate Chamfer Metrics reconstr_img_scaled <-> pcl_gt_scaled
	pcl_gt_scaled, reconstr_img_scaled = scale(pcl_gt, reconstr_img)
	dists_forward_rimg_scaled, dists_backward_rimg_scaled, chamfer_distance_rimg_scaled = [tf.reduce_mean(metric) for metric in get_chamfer_metrics(pcl_gt_scaled, reconstr_img_scaled)]

	# Calculate emd on scaled prediction and GT
	emd = tf.reduce_mean(get_emd_metrics(pcl_gt_scaled, reconstr_img_scaled, BATCH_SIZE, GT_PCL_SIZE))

	# L1 Distance between latent representations
	#L1 = tf.reduce_mean(tf.abs(z_latent_pcl - z_latent_img))

	# L2 Distance between latent representations
	#L2 = tf.reduce_mean((z_latent_pcl - z_latent_img)**2)

	# Define Loss to optimize on
	if FLAGS.loss == 'chamfer':
		loss = chamfer_distance_rimg_scaled
	elif FLAGS.loss == 'cd_emd':
		loss = chamfer_distance_rimg_scaled + emd
	elif FLAGS.loss == 'emd':
		loss = emd


	# Get Training Models
	train_models, val_models, train_pair_indices, val_pair_indices = get_shapenet_models(FLAGS)
	batches = len(train_pair_indices) / BATCH_SIZE

	# Get training vars and pointnet_ae vars
	psgn_vars = [var for var in tf.global_variables() if 'psgn' in var.name]
	pu_gen_vars = [var for var in tf.global_variables() if 'generator' in var.name]
	train_vars = psgn_vars + pu_gen_vars

	# Define Optimizer
	optim = tf.train.AdamOptimizer(FLAGS.lr, beta1=0.9).minimize(loss, var_list=train_vars)

	start_epoch = 0
	max_epoch = FLAGS.max_epoch

	# Define Log Directories
	snapshot_folder = join(FLAGS.exp, 'snapshots')
	best_folder = join(FLAGS.exp, 'best')
	logs_folder = join(FLAGS.exp, 'logs')
	pointnet_ae_logs_path = FLAGS.ae_logs

	# Define Savers
	saver = tf.train.Saver(max_to_keep=2)

	# Define Summary Placeholders
	ph_dists_forward = tf.placeholder(tf.float32, name='dists_forward') 
	ph_dists_backward = tf.placeholder(tf.float32, name='dists_backward') 
	ph_dists_chamfer = tf.placeholder(tf.float32, name='dists_chamfer') 
	ph_l1 = tf.placeholder(tf.float32, name='l1') 
	ph_l2 = tf.placeholder(tf.float32, name='l2') 
	ph_loss = tf.placeholder(tf.float32, name='loss')
	ph_emd = tf.placeholder(tf.float32, name='emd') 

	ph_summary = [ph_dists_forward, ph_dists_backward, ph_dists_chamfer, ph_l1, ph_l2, ph_loss, ph_emd]
	merged_summ = get_summary(ph_summary)

	# Create log directories
	create_folders([snapshot_folder, logs_folder, join(snapshot_folder, 'best'), best_folder])

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	with tf.Session(config=config) as sess:

		train_writer = tf.summary.FileWriter(logs_folder+'/train', sess.graph_def)
		val_writer = tf.summary.FileWriter(logs_folder+'/val', sess.graph_def)

		sess.run(tf.global_variables_initializer())

		# Load pretrained frozen pointnet ae weights
		load_pointnet_ae(pointnet_ae_logs_path, pu_gen_vars, sess, FLAGS)

		# Load previous checkpoint
		start_epoch = load_previous_checkpoint(snapshot_folder, saver, sess)

		best_val_loss = 10000000
		since = time.time()

		print '*'*30,'\n','Training Started !!!\n', '*'*30

		PRINT_N = FLAGS.print_n

		if FLAGS.sanity_check:
			random.shuffle(train_pair_indices)
			from utils.show_3d import show3d_balls
			for b in xrange(batches):
				batch_ip, batch_gt = fetch_batch(train_models, train_pair_indices, b, BATCH_SIZE)
				result = sess.run(reconstr_pcl, feed_dict={pcl_gt:batch_gt})
				for ind in xrange(BATCH_SIZE):
					show3d_balls.showtwopoints(batch_gt[ind], result[ind], ballradius=3)

		for i in xrange(start_epoch, max_epoch):
			random.shuffle(train_pair_indices)
			stats = {}
			stats = reset_stats(ph_summary, stats)
			iter_start = time.time()

			for b in xrange(batches):
				global_step = i*batches + b + 1
				batch_ip, batch_gt_den = fetch_batch_dense(train_models, train_pair_indices, b, BATCH_SIZE)
				runlist = [loss, optim]
				_l, _ = sess.run(runlist, feed_dict={pcl_gt:batch_gt_den, img_inp:batch_ip})
				_summary_losses = [_l]
				stats = update_stats(ph_summary, _summary_losses, stats, PRINT_N)

				if global_step % PRINT_N == 0:
					summ = sess.run(merged_summ, feed_dict=stats)
					train_writer.add_summary(summ, global_step)
					till_now = time.time() - iter_start
					print 'Loss = {} Iter = {}  Minibatch = {} Time:{:.0f}m {:.0f}s'.format(
						stats[ph_loss], global_step, b, till_now//60, till_now%60
					)
					stats = reset_stats(ph_summary, stats)
					iter_start = time.time()

			print 'Saving Model ....................'
			saver.save(sess, join(snapshot_folder, 'model'), global_step=i)
			print '..................... Model Saved'

			val_epoch_chamfer, val_epoch_forward, val_epoch_backward, val_epoch_emd, val_summ = get_epoch_loss(val_models, val_pair_indices)
			val_writer.add_summary(val_summ, global_step)

			time_elapsed = time.time() - since

			print '-'*65 + ' EPOCH ' + str(i) + ' ' + '-'*65
			print 'Val Chamfer: {:.8f}  Forward: {:.8f}  Backward: {:.8f} Emd: {:.8f} Time:{:.0f}m {:.0f}s'.format(
				val_epoch_chamfer, val_epoch_forward, val_epoch_backward, val_epoch_emd, time_elapsed//60, time_elapsed%60
			)
			print '-'*140
			print

			if (val_epoch_chamfer < best_val_loss):
				print 'Saving Best at Epoch %d ...............'%(i)
				saver.save(sess, join(snapshot_folder, 'best', 'best'))
				os.system('cp %s %s'%(join(snapshot_folder, 'best/*'), best_folder))
				best_val_loss = val_epoch_chamfer
				print '.............................Saved Best'