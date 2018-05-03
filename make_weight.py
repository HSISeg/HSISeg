import json,os,sys,traceback,django
from algo import pdhg_weight
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "HSISeg.settings")
django.setup()
from algo.models import Results
params = json.loads(sys.argv[1])
pid = os.getpid()
try:
	pid_element = Results.objects.get(id=params.get('id'))
	pid_element.pid = pid
	pid_element.save()
except Results.DoesNotExist as e:
	pid_element = Results.objects.create(pid=pid)
try:
	from algo import image_helper as ih
	from algo import algo_default_params as default_params
	weight_algo = params.get('algo')
	image_pickle_file_path = params.get("image_pickle_file_path")
	image = ih.get_pickle_object_as_numpy(image_pickle_file_path)
	maxconn = default_params.gen_default_params['maxconn'] if not params.get('maxconn') else params.get('maxconn') 
	output_path = params.get('output_path')
	if not weight_algo:
		weight_algo = default_params.default_weight_algo
	weight_algo_params = params.get('algo_params')
	from algo import pdhg_weight
	weight_algo_func =  getattr(pdhg_weight,weight_algo)
	weight = weight_algo_func(image,weight_algo_params)
	ih.save_to_pickle(weight,output_path+"/weight.pickle")
	pid_element.is_done = True
	pid_element.percentage_done = '100'
	pid_element.status_text = 'Success'
	pid_element.result_file_name = output_path+"/weight.pickle"
	pid_element.save()
except Exception as e:
	pid_element.is_done = True
	pid_element.percentage_done = '100'
	pid_element.status_text = 'Failed'
	pid_element.error = str(traceback.format_exc())
	pid_element.save()

