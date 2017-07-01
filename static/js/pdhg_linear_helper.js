$('#run_algorithm').on('click', function (e) {

     var request_data_json = construct_api_request_object();
     start_algorithm(request_data_json);
})


function append_to_log(append_text) {
	var $api_log = $('#api_log');
	$api_log.append(append_text + "\n********************\n");
	$api_log.scrollTop($api_log[0].scrollHeight);
	return
}

function construct_api_request_object() {
	var request_data = new Object();
	request_data.image_pickle_file_path = $("#image_pickle_file_path").val();
	request_data.output_path = $("#output_path").val();
	request_data.algo = "linear_pdhg";
	request_data.algo_params = new Object();
	request_data.algo_params.sigma = parseFloat($("#sigma").val());
	request_data.algo_params.mu = parseFloat($("#mu").val());
	request_data.algo_params.lamda = parseFloat($("#lamda").val());
	request_data.algo_params.tao = parseFloat($("#tao").val());
	request_data.algo_params.theta = parseFloat($("#theta").val());
	request_data.algo_params.iter_stop = parseFloat($("#iter_stop").val());
	request_data.algo_params.innerloop = parseInt($("#innerloop").val());
	request_data.algo_params.outerloop = parseInt($("#outerloop").val());
	request_data.cluster_number = parseInt($("#cluster_number").val());
	request_data.weight_pickle_file_path = $("#weight_pickle_file_path").val();
	request_data.centroid_pickle_file_path = $("#centroid_pickle_file_path").val();
	request_data.maxconn = parseInt($("#maxconn").val());


	request_data_json = JSON.stringify(request_data);


	append_to_log("Request JSON generated\n" + request_data_json);
	update_cache_input_variables();
  	return request_data_json;
}

function update_cache_input_variables(){
	localStorage['image_pickle_file_path'] = $("#image_pickle_file_path").val();
	localStorage['output_path'] = $("#output_path").val();
	return;
}

function start_algorithm(request_data_json) {
	$.post("/run-algo",request_data_json,start_algorithm_callback,"text");
	return;
}

function start_algorithm_callback(data,status) {
	append_to_log("Response For Algorithm Call")
	append_to_log("[" + status + "] " + data);
	var data = JSON.parse(data);
	localStorage['pdhg_linear_task_id'] = data.task_id;
	$('#task_id').text(data.task_id);
	poll_for_task(data.task_id);
	return;
}


function poll_for_task(task_id) {
	var poll_request = new Object();
	poll_request.task_id = task_id;
	$.post("/get-task-status",JSON.stringify(poll_request),poll_for_task_callback,"text");
	return;
}

function poll_for_task_callback(data,status) {
	append_to_log("Result of Task Poll")
	append_to_log("[" + status + "] " + data);
	var data = JSON.parse(data);
	$('#pid').text(data.pid);
	localStorage['pdhg_linear_pid'] = data.pid;
	update_progress_bar(data.percentage_done,false);
	if (!data.is_done){
		setTimeout(function(){ poll_for_task(data.task_id); }, 2000);
	}else{
		update_progress_bar(100,false);
	}
	return;
}

function update_progress_bar(percentage_done,fake_percentage) {
	if(!fake_percentage){
		$("#progress_bar").css("width", percentage_done + "%");
	}else{
		var current_progress = parseFloat($("#progress_bar").width());
		var total_progress = parseFloat($("#progress").width());
		var current_fraction = current_progress / total_progress;
		var new_progress = (1 - (1 - current_fraction)  / Math.pow(Math.E,0.4) ) * 90;
		$("#progress_bar").css("width", new_progress + "%");
	}
	return;
}

function init() {
	$("#image_pickle_file_path").val(localStorage['image_pickle_file_path'] || '');
	$("#output_path").val(localStorage['output_path'] || '');
	$("#cluster_number").val(localStorage['cluster_number'] || '');
	$("#centroid_pickle_file_path").val(localStorage['centroid_pickle_file_path'] || '');
	$("#weight_pickle_file_path").val(localStorage['weight_pickle_file_path'] || '')
	$('#pid').text(localStorage['pdhg_linear_pid'] || '');
	$('#task_id').text(localStorage['pdhg_linear_task_id'] || '');
}