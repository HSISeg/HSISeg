from django.shortcuts import render
from django.http import HttpResponse, Http404, HttpResponseForbidden, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from django import forms
from django.shortcuts import render_to_response
from django.template import RequestContext
from . import algo_default_params as default_params
import json,os,subprocess
from algo.image_helper import get_data_from_image,save_to_pickle
from algo.models import Results
# import psutil

@csrf_exempt
def image_to_pickle(request):
	if request.method == 'POST':
		try:
			params = json.loads(request.body)
		except:
			return HttpResponseBadRequest(json.dumps({'error':'Json required'}),content_type="application/json")
		if not params.get('image_path') or not params.get('output_path'):
			return HttpResponseBadRequest(json.dumps({'error':'image_path and output_path required'}),content_type="application/json")
		try:
			data = get_data_from_image(params.get('image_path'))
			save_to_pickle(data,str(params.get('output_path'))+"/data.pickle")
			return HttpResponse(json.dumps({'success':True}),content_type="application/json")
		except Exception as e:
			return HttpResponseBadRequest(json.dumps({'error':str(e)}),content_type="application/json")
	else:
		raise Http404()

@csrf_exempt
def get_algo_params(request):
	if request.method == 'GET':
		algo = request.GET.get('algo')
		if not algo:
			return HttpResponseBadRequest(json.dumps({'error':'algo mandatory'}),content_type="application/json")
		return HttpResponse(json.dumps(default_params.algo_details.get(algo)),content_type="application/json")
	else:
		raise Http404()

@csrf_exempt
def get_algo_names(request):
	return HttpResponse(json.dumps({'algo':list(default_params.algo_details.keys())}),content_type="application/json")

@csrf_exempt
def home(request):
	return render_to_response('home.html', {}, content_type='text/html')

@csrf_exempt
def pdhg_linear_ui(request):
	return render_to_response('pdhg_linear_ui.html', {}, content_type='text/html')

@csrf_exempt
def initialization(request):
	return render_to_response('initialization.html', {}, content_type='text/html')

@csrf_exempt
def weight_calculation(request):
	return render_to_response('weight_calculation.html', {}, content_type='text/html')

@csrf_exempt
def quad_pdhg_ui(request):
	return render_to_response('quad_pdhg_ui.html', {}, content_type='text/html')

@csrf_exempt
def beta_calculation(request):
	return render_to_response('beta_calculation.html', {}, content_type='text/html')

@csrf_exempt
def fuzzy_c_ui(request):
	return render_to_response('fuzzy_c_ui.html', {}, content_type='text/html')

@csrf_exempt
def run_algo(request):
	if request.method == 'POST':
		try:
			params = json.loads(request.body)
		except:
			return HttpResponseBadRequest(json.dumps({'error':'Json required'}),content_type="application/json")
		if not params.get("image_pickle_file_path") or not params.get("algo") or not params.get('cluster_number') or not params.get('output_path'): 
			return HttpResponseBadRequest(json.dumps({'error':'output_path ,image_pickle_file_path,cluster_number and algo manadatory,only .pickle file accepted, valid algo are '+str(list(default_params.algo_details.keys()))}),content_type="application/json")
		pid_element = Results.objects.create(pid=None)
		params['id'] = pid_element.id
		params = json.dumps(params)
		command = "python segment_hsi.py '"+str(params)+"' &"
		proc = subprocess.Popen(command,shell=True)
		return HttpResponse(json.dumps({'success':True,'task_id':pid_element.id}),content_type="application/json")
	else:
		raise Http404()

@csrf_exempt
def get_beta(request):
	if request.method == 'POST':
		try:
			params = json.loads(request.body)
		except:
			return HttpResponseBadRequest(json.dumps({'error':'Json required'}),content_type="application/json")
		if not params.get("output_path") or not params.get("image_pickle_file_path"): 
			return HttpResponseBadRequest(json.dumps({'error':'output_path and image_pickle_file_path manadatory'}),content_type="application/json")
		pid_element = Results.objects.create(pid=None)
		params['id'] = pid_element.id
		params = json.dumps(params)
		command = "python make_beta.py '"+str(params)+"' &"
		proc = subprocess.Popen(command,shell=True)
		return HttpResponse(json.dumps({'success':True,'task_id':pid_element.id}),content_type="application/json")
	else:
		raise Http404()

@csrf_exempt
def get_weight(request):
	if request.method == 'POST':
		try:
			params = json.loads(request.body)
		except:
			return HttpResponseBadRequest(json.dumps({'error':'Json required'}),content_type="application/json")
		if not params.get("output_path") or not params.get("image_pickle_file_path"): 
			return HttpResponseBadRequest(json.dumps({'error':'output_path and image_pickle_file_path manadatory'}),content_type="application/json")
		pid_element = Results.objects.create(pid=None)
		params['id'] = pid_element.id
		params = json.dumps(params)
		command = "python make_weight.py '"+str(params)+"' &"
		proc = subprocess.Popen(command,shell=True)
		return HttpResponse(json.dumps({'success':True,'task_id':pid_element.id}),content_type="application/json")
	else:
		raise Http404()

@csrf_exempt
def get_initial_centroid(request):
	if request.method == 'POST':
		try:
			params = json.loads(request.body)
		except:
			return HttpResponseBadRequest(json.dumps({'error':'Json required'}),content_type="application/json")
		if not params.get("output_path") or not params.get("image_pickle_file_path") or not params.get('cluster_number'): 
			return HttpResponseBadRequest(json.dumps({'error':'output_path,cluster_number and image_pickle_file_path manadatory'}),content_type="application/json")
		pid_element = Results.objects.create(pid=None)
		params['id'] = pid_element.id
		params = json.dumps(params)
		command = "python make_centroid.py '"+str(params)+"' &"
		proc = subprocess.Popen(command,shell=True)
		return HttpResponse(json.dumps({'success':True,'task_id':pid_element.id}),content_type="application/json")
	else:
		raise Http404()

@csrf_exempt
def kill_task(request):
	if request.method == 'POST':
		try:
			params = json.loads(request.body)
		except:
			return HttpResponseBadRequest(json.dumps({'error':'Json required'}),content_type="application/json")
		if not params.get("task_id"): 
			return HttpResponseBadRequest(json.dumps({'error':'task_id manadatory'}),content_type="application/json")
		tasks = Results.objects.filter(id = params['task_id'])
		for task in tasks:
			try:
				import psutil
				parent = psutil.Process(task.pid)
				for child in parent.children(recursive=True):
				    child.kill()
				parent.kill()
			except Exception as e:
				pass
			task.status_text = 'Killed'
			task.is_done = True
			task.save()
		return HttpResponse(json.dumps({'success':True}),content_type="application/json")
	else:
		raise Http404()

@csrf_exempt
def get_task_status(request):
	if request.method == 'POST':
		try:
			params = json.loads(request.body)
		except:
			return HttpResponseBadRequest(json.dumps({'error':'Json required'}),content_type="application/json")
		if not params.get("task_id"): 
			return HttpResponseBadRequest(json.dumps({'error':'task_id manadatory'}),content_type="application/json")
		try:
			task = Results.objects.get(id=params['task_id'])
			
			data = {'task_id':task.id,'result_file_name':task.result_file_name,'error':task.error,'percentage_done':task.percentage_done,
					'status_text':task.status_text,'is_done':task.is_done,'pid':task.pid}
			return HttpResponse(json.dumps(data),content_type="application/json")
		except Results.DoesNotExist as e:
			return HttpResponseBadRequest(json.dumps({'error':'Invalid task_id'}),content_type="application/json")
	else:
		raise Http404()

		