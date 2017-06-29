from django.shortcuts import render
from django.http import HttpResponse, Http404, HttpResponseForbidden, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from django import forms
from django.shortcuts import render_to_response
from django.template import RequestContext
import algo_default_params as default_params
import json,os,subprocess
from algo.models import Results


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
	return HttpResponse(json.dumps({'algo':default_params.algo_details.keys()}),content_type="application/json")

@csrf_exempt
def home(request):
	return render_to_response('home.html', {}, content_type='text/html')

@csrf_exempt
def run_algo(request):
	if request.method == 'POST':
		try:
			params = json.loads(request.body)
		except:
			return HttpResponseBadRequest(json.dumps({'error':'Json required'}),content_type="application/json")
		if not params.get("image_pickle_file_path") or not params.get("algo") or not params.get('cluster_number') or not params.get('output_path'): 
			return HttpResponseBadRequest(json.dumps({'error':'output_path ,image_pickle_file_path,cluster_number and algo manadatory,only .pickle file accepted, valid algo are '+str(default_params.algo_details.keys())}),content_type="application/json")
		pid_element = Results.objects.create(task_id=None)
		params['id'] = pid_element.id
		params = json.dumps(params)
		command = "python segment_hsi.py '"+str(params)+"' &"
		proc = subprocess.Popen(command,shell=True)
		return HttpResponse(json.dumps({'success':True,'pid':pid_element.id}),content_type="application/json")
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
		pid_element = Results.objects.create(task_id=None)
		params['id'] = pid_element.id
		params = json.dumps(params)
		command = "python make_beta.py '"+str(params)+"' &"
		proc = subprocess.Popen(command,shell=True)
		return HttpResponse(json.dumps({'success':True,'pid':pid_element.id}),content_type="application/json")
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
		pid_element = Results.objects.create(task_id=None)
		params['id'] = pid_element.id
		params = json.dumps(params)
		command = "python make_weight.py '"+str(params)+"' &"
		proc = subprocess.Popen(command,shell=True)
		return HttpResponse(json.dumps({'success':True,'pid':pid_element.id}),content_type="application/json")
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
		pid_element = Results.objects.create(task_id=None)
		params['id'] = pid_element.id
		params = json.dumps(params)
		command = "python make_centroid.py '"+str(params)+"' &"
		proc = subprocess.Popen(command,shell=True)
		return HttpResponse(json.dumps({'success':True,'pid':pid_element.id}),content_type="application/json")
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
		try:
			tasks = Results.objects.filter(id = params['task_id'],is_done=False)
			for task in tasks:
				command = "kill -9 "+str(task.task_id)
				proc = subprocess.Popen(command,shell=True,stdout=file("1.txt", "ab"))
				task.status_text = 'Killed'
				task.is_done = True
				task.save()
			return HttpResponse(json.dumps({'success':True}),content_type="application/json")

		except Exception as e:
			return HttpResponseBadRequest(json.dumps({'success':False,'error':str(e)}),content_type="application/json")
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
		tasks = Results.objects.filter(id=params['task_id'])
		result = []
		for task in tasks:
			data = {'task_id':task.task_id,'result_file_name':task.result_file_name,'error':task.error,'percentage_done':task.percentage_done,
					'status_text':task.status_text,'is_done':task.is_done}
			result.append(data)
		return HttpResponse(json.dumps(result),content_type="application/json")
	else:
		raise Http404()

		