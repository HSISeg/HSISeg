import json,os,sys,traceback
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "HSISeg.settings")
from algo.models import Results
params = json.loads(sys.argv[1])
pid = os.getpid()
try:
	pid_element = Results.objects.get(id=params.get('id'))
	pid_element.task_id = pid
	pid_element.save()
except Results.DoesNotExist as e:
	pid_element = Results.objects.create(task_id=pid)
try:
	from algo import image_helper as ih
	from algo import algo_default_params as default_params
	import importlib
	algo = params.get('algo')
	image_pickle_file_path = params.get("image_pickle_file_path")
	image = ih.get_pickle_object_as_numpy(image_pickle_file_path)
	maxconn = default_params.gen_default_params['maxconn'] if not params.get('maxconn') else params.get('maxconn') 
	output_path = params.get('output_path')
	output_path += '/output'
	file = importlib.import_module(default_params.algo_main_func[algo]['file'])
	main_func =  getattr(file,default_params.algo_main_func[algo]['func'])
	main_func(image,params.get('cluster_number'),output_path,maxconn,params,pid_element)
	pid_element.is_done = True
	pid_element.percentage_done = '100'
	pid_element.status_text = 'Success'
	pid_element.result_file_name = str(output_path)+'*.jpeg'
	pid_element.save()
except Exception as e:
	pid_element.is_done = True
	pid_element.percentage_done = '100'
	pid_element.error = str(traceback.format_exc())
	pid_element.status_text = 'Failed'
	pid_element.save()

