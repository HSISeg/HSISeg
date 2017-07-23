import json,os,sys,traceback,django
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
	from algo.image_helper import get_data_from_image,save_to_pickle
	data = get_data_from_image(params.get('image_file_path'))
	save_to_pickle(data,str(params.get('output_path'))+"/data.pickle")
	pid_element.is_done = True
	pid_element.percentage_done = '100'
	pid_element.status_text = 'Success'
	pid_element.result_file_name = str(params.get('output_path'))+"/data.pickle"
	pid_element.save()
except Exception as e:
	pid_element.is_done = True
	pid_element.percentage_done = '100'
	pid_element.error = str(traceback.format_exc())
	pid_element.status_text = 'Failed'
	pid_element.save()

