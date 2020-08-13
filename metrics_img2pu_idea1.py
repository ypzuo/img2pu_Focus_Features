from importer import *
from utils.icp import icp
from tqdm import tqdm
from utils.encoder_2 import *

parser = argparse.ArgumentParser()

# Machine Details
parser.add_argument('--gpu', type=str, required=True, help='[Required] GPU to use')

# Dataset
#parser.add_argument('--dataset', type=str, required=True, help='Choose from [shapenet, pix3d]')
parser.add_argument('--data_dir_imgs', type=str, required=True, help='Path to shapenet rendered images')
parser.add_argument('--data_dir_pcl', type=str, required=True, help='Path to shapenet pointclouds')

# Experiment Details
#parser.add_argument('--mode', type=str, required=True, help='[Required] Latent Matching setup. Choose from [lm, plm]')
parser.add_argument('--exp', type=str, required=True, help='[Required] Path of experiment for loading pre-trained model')
parser.add_argument('--category', type=str, required=True, help='[Required] Model Category for training')
parser.add_argument('--load_best', action='store_true', help='load best val model')

# AE Details
parser.add_argument('--bottleneck', type=int, required=False, default=512, help='latent space size')
parser.add_argument('--up_ratio', type=int, default=2, help='up sampling ratio')
# parser.add_argument('--bn_encoder', action='store_true', help='Supply this parameter if you want bn_encoder, otherwise ignore')
parser.add_argument('--bn_decoder', action='store_true', help='Supply this parameter if you want bn_decoder, otherwise ignore')
parser.add_argument('--bn_decoder_final', action='store_true', help='Supply this parameter if you want bn_decoder, otherwise ignore')

# Fetch Batch Details
parser.add_argument('--batch_size', type=int, default=24, help='Batch Size during evaluation')
parser.add_argument('--eval_set', type=str, help='Choose from train/valid')

# Other Args
parser.add_argument('--visualize', action='store_true', help='supply this parameter to visualize')

FLAGS = parser.parse_args()

print '-='*50
print FLAGS
print '-='*50

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

BATCH_SIZE = FLAGS.batch_size
if FLAGS.visualize:
	BATCH_SIZE = 1
NUM_POINTS = 2048
NUM_EVAL_POINTS = 2048
GT_PCL_SIZE = 2048
NUM_VIEWS = 24
HEIGHT = 128
WIDTH = 128
UP_RATIO = FLAGS.up_ratio

if FLAGS.visualize:
	from utils.show_3d import show3d_balls
	ballradius = 3

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
		pcl_path = join(FLAGS.data_dir_pcl, model_path, 'pointcloud_2048.npy')

		pcl_gt = np.load(pcl_path)

		ip_image = cv2.imread(img_path)[4:-5, 4:-5, :3]
		ip_image = cv2.cvtColor(ip_image, cv2.COLOR_BGR2RGB)

		batch_gt.append(pcl_gt)
		batch_ip.append(ip_image)

	return np.array(batch_ip), np.array(batch_gt)

def fetch_batch_pix3d(models, batch_num, batch_size):
	''' 
	Inputs:
		models: List of pix3d dicts
		batch_num: batch_num during epoch
		batch_size: batch size for training or validation
	Returns:
		batch_ip: input RGB image of shape (B, HEIGHT, WIDTH, 3)
		batch_gt: gt point cloud of shape (B, NUM_POINTS, 3)
	Description:
		Batch Loader for Pix3D dataset
	'''
	batch_ip = []
	batch_gt = []

	for ind in xrange(batch_num*batch_size,batch_num*batch_size+batch_size):
		_dict = models[ind]
		model_path = '/'.join(_dict['model'].split('/')[:-1])
		model_name = re.search('model(.*).obj', _dict['model'].strip().split('/')[-1]).group(1)
		img_path = join(FLAGS.data_dir_imgs, _dict['img'])
		mask_path = join(FLAGS.data_dir_imgs, _dict['mask'])
		bbox = _dict['bbox'] # [width_from, height_from, width_to, height_to]
		pcl_path_1K = join(FLAGS.data_dir_pcl, model_path,'pcl_%d%s.npy'%(NUM_EVAL_POINTS,model_name))
		ip_image = cv2.imread(img_path)
		ip_image = cv2.cvtColor(ip_image, cv2.COLOR_BGR2RGB)
		mask_image = cv2.imread(mask_path)!=0
		ip_image=ip_image*mask_image
		ip_image = ip_image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]

		current_size = ip_image.shape[:2] # current_size is in (height, width) format
		ratio = float(HEIGHT-PAD)/max(current_size)
		new_size = tuple([int(x*ratio) for x in current_size])
		ip_image = cv2.resize(ip_image, (new_size[1], new_size[0])) # new_size should be in (width, height) format
		delta_w = WIDTH - new_size[1]
		delta_h = HEIGHT - new_size[0]
		top, bottom = delta_h//2, delta_h-(delta_h//2)
		left, right = delta_w//2, delta_w-(delta_w//2)
		color = [0, 0, 0]
		ip_image = cv2.copyMakeBorder(ip_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

		xangle = np.pi/180. * -90
		yangle = np.pi/180. * -90
		pcl_gt = rotate(rotate(np.load(pcl_path_1K), xangle, yangle), xangle)

		batch_gt.append(pcl_gt)
		batch_ip.append(ip_image)

	return np.array(batch_ip), np.array(batch_gt)

def calculate_metrics(models, indices, pcl_gt_1K_scaled, pred_scaled):

	batches = len(indices)/BATCH_SIZE
	if FLAGS.visualize:
		iters = range(batches)
	else:
		iters = tqdm(range(batches))

	epoch_chamfer = 0.
	epoch_forward = 0.
	epoch_backward = 0.
	epoch_emd = 0.	

	ph_gt = tf.placeholder(tf.float32, (BATCH_SIZE, NUM_EVAL_POINTS, 3), name='ph_gt')
	ph_pr = tf.placeholder(tf.float32, (BATCH_SIZE, NUM_EVAL_POINTS, 3), name='ph_pr')

	dists_forward, dists_backward, chamfer_distance = get_chamfer_metrics(ph_gt, ph_pr)
	emd = get_emd_metrics(ph_gt, ph_pr, BATCH_SIZE, NUM_EVAL_POINTS)

	for cnt in iters:
		start = time.time()

		batch_ip, batch_gt_1K = fetch_batch(models, indices, cnt, BATCH_SIZE)

		_gt_scaled_1K, _pr_scaled = sess.run(
			[pcl_gt_1K_scaled, pred_scaled], 
			feed_dict={pcl_gt_1K:batch_gt_1K, img_inp:batch_ip}
		)

		_pr_scaled_icp = []

		for i in xrange(BATCH_SIZE):
			rand_indices = np.random.permutation(NUM_POINTS)[:NUM_EVAL_POINTS]
			T, _, _ = icp(_gt_scaled_1K[i], _pr_scaled[i][rand_indices], tolerance=1e-10, max_iterations=1000)
			_pr_scaled_icp.append(np.matmul(_pr_scaled[i][rand_indices], T[:3,:3]) - T[:3, 3])

		_pr_scaled_icp = np.array(_pr_scaled_icp).astype('float32')

		C,F,B,E = sess.run(
			[chamfer_distance, dists_forward, dists_backward, emd], 
			feed_dict={ph_gt:_gt_scaled_1K, ph_pr:_pr_scaled_icp}
		)

		epoch_chamfer += C.mean() / batches
		epoch_forward += F.mean() / batches
		epoch_backward += B.mean() / batches
		epoch_emd += E.mean() / batches

		if FLAGS.visualize:
			for i in xrange(BATCH_SIZE):
				print '-'*50
				print C[i], F[i], B[i], E[i]
				print '-'*50
				#ip = np.where(batch_ip[i]>0,batch_ip[i],255)
				cv2.imshow('', batch_ip[i])
				print 'Displaying Gt scaled icp 1k'
				show3d_balls.showpoints(_gt_scaled_1K[i], ballradius=3)
				print 'Displaying Pr scaled icp 1k'
				show3d_balls.showpoints(_pr_scaled_icp[i], ballradius=3)
		
		if cnt%10 == 0:
			print '%d / %d' % (cnt, batches)

	if not FLAGS.visualize:
		log_values(csv_path, epoch_chamfer, epoch_forward, epoch_backward, epoch_emd)

	return 



if __name__ == '__main__':

	# Create Placeholders
	img_inp = tf.placeholder(tf.float32, shape=(BATCH_SIZE, HEIGHT, WIDTH, 3), name='img_inp')
	pcl_gt_1K = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_EVAL_POINTS, 3), name='pcl_gt_1K')

	is_training = False
	bn_decay = 0.95
	# Generate Prediction
	with tf.variable_scope('psgn') as scope:
		z_latent_img = image_encoder_se_pure(img_inp, FLAGS)

		out_img = decoder_with_fc_only(z_latent_img, layer_sizes=[512,1024,np.prod([1024, 3])],
			b_norm=FLAGS.bn_decoder,
			b_norm_finish=False,
			verbose=True,
			scope=scope
			)
	reconstr = tf.reshape(out_img, (BATCH_SIZE, 1024, 3))
	is_training = True
	bn_decay = 0.95

	features = feature_extraction(reconstr, scope='feature_extraction', is_training=True, bn_decay=None)

	with tf.variable_scope('generator') as scope:
		up_l2_points, l1_points, l2_points = get_local_features_with_sift(reconstr, is_training=True, scope=scope,
						reuse=None, use_normal=False, use_bn=False, use_ibn=False,
						bn_decay=bn_decay,up_ratio=UP_RATIO)

	with tf.variable_scope('up_layer') as scope:
		net = concat_features(features, up_l2_points, l1_points, l2_points, is_training, scope, reuse=None, 
				bn_decay=None,)


	coord = conv2d(net, 64, [1, 1],
					padding='VALID', stride=[1, 1],
					bn=False, is_training=True,
					scope='fc_layer1', bn_decay=None)

	coord = conv2d(coord, 3, [1, 1],
					padding='VALID', stride=[1, 1],
					bn=False, is_training=True,
					scope='fc_layer2', bn_decay=None,
					activation_fn=None, weight_decay=0.0)
	outputs = tf.squeeze(coord, [2])
	outputs = sample(2048, outputs)  #equals to :outputs = gather_point(outputs, farthest_point_sample(2048, outputs)) 
	out = tf.reshape(outputs, (BATCH_SIZE, GT_PCL_SIZE, 3))

	# Perform Scaling
	pcl_gt_1K_scaled, reconstr_img_scaled = scale(pcl_gt_1K, out)

	# Snapshot Folder Location
	if FLAGS.load_best:
		snapshot_folder = join(FLAGS.exp, 'best')
	else:
		snapshot_folder = join(FLAGS.exp, 'snapshots')

 	# Metrics path
 	metrics_folder = join(FLAGS.exp, 'metrics', FLAGS.eval_set)
 	create_folder(metrics_folder)
	csv_path = join(metrics_folder,'%s.csv'%FLAGS.category)
	with open(csv_path, 'w') as f:
		f.write('Chamfer, Fwd, Bwd, Emd\n')

	# GPU configuration
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	with tf.Session(config=config) as sess:

		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()

		# Load previous checkpoint
		load_previous_checkpoint(snapshot_folder, saver, sess, is_training=False)

		tflearn.is_training(False, session=sess)

		train_models, val_models, train_pair_indices, val_pair_indices = get_shapenet_models(FLAGS)

		if FLAGS.visualize:
			random.shuffle(val_pair_indices)
			random.shuffle(train_pair_indices)

		if FLAGS.eval_set == 'train':
			calculate_metrics(train_models, train_pair_indices, pcl_gt_1K_scaled, reconstr_img_scaled)
		if FLAGS.eval_set == 'valid':
			calculate_metrics(val_models, val_pair_indices, pcl_gt_1K_scaled, reconstr_img_scaled)
