fuzzy_c_means_beta_algo = {'beta_spectral_spatial':{
												'algo_params':{'max_points':10,'half_search_window':10,'half_patch_size':1, 'gaussian_sigma':3,'euclidean_distance_weight': 0 ,'theta':0.7}},
						'beta_spatial':{
									'algo_params':{'theta':0.7,'max_points':500}}}
default_beta_algo = 'beta_spatial'

cluster_number = 17

default_weight_algo = 'pdhg_sparse_weight'

default_centroid_init_algo = 'scklearn_kmeans_centroids'

default_cluster_algo = 'linear_pdhg'

centroid_init_algo = {'scklearn_kmeans_centroids':{},'scipy_kmeans_centroids':{},'random_centroids':{}}

weight_algo = {'pdhg_sparse_weight':
							{'algo_params':
											{'max_points':10,'half_search_window':10,'half_patch_size':1, 'gaussian_sigma':3,'euclidean_distance_weight': 0}}}

algo_details = {'linear_pdhg':{'algo_params':{'sigma':0.0008333333333333334,'mu': 1e-5,'lamda':1e6,'tao':10.0,'theta':1.0,'iter_stop':1.1,'innerloop': 5,'outerloop': 50},
								'preprocessing':{
									'weight':{ 'default_algo':'pdhg_sparse_weight','options':weight_algo},
									'centroid_init':{'default_algo':'scipy_kmeans_centroids','options':centroid_init_algo}}
								},

				'quad_pdhg':{'algo_params':{'sigma':0.0008333333333333334,'mu': 1e-5,'lamda':1e6,'tao':10.0,'theta':1.0,'iter_stop':1.1,'innerloop': 10,'outerloop': 10},
							'preprocessing':{
								'weight':{ 'default_algo':'pdhg_sparse_weight','options':weight_algo},
								'centroid_init':{'default_algo':'scipy_kmeans_centroids','options':centroid_init_algo}}
							},
				'fuzzy_c_means':{'algo_params':{'fuzzy_index':1.3,'terminating_mean_error':0.0002,'alpha_w':9.0,'alpha_e_avg_t':0.001,'alpha_n0':20,'max_iter':50},
								'preprocessing':{
									'centroid_init':{'default_algo':'scipy_kmeans_centroids','options':centroid_init_algo},
									'beta':{'default_algo':'beta_spatial','options':fuzzy_c_means_beta_algo}}
								}
							}

gen_default_params = {'maxconn':8,'algo':'quad_pdhg'}

algo_main_func = {'linear_pdhg':{'file':'algo.linear_pdhg','func':'run_pdhg_linear'},
				'quad_pdhg':{'file':'algo.quad_pdhg','func':'run_pdhg_quad'},
				'fuzzy_c_means':{'file':'algo.fuzzy_c_means','func':'run_fuzzy_c_means'}}
