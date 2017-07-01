$('#run_algorithm').on('click', function (e) {
	var request_data_json = construct_api_request_object();
	start_algorithm(request_data_json);
})


function construct_api_request_object() {

	var request_data = new Object();
	request_data.image_pickle_file_path = $("#image_pickle_file_path").val();
	request_data.output_path = $("#output_path").val();
	request_data.algo = "fuzzy_c_means";
	request_data.algo_params = new Object();
	request_data.algo_params.fuzzy_index = parseFloat($("#fuzzy_index").val());
	request_data.algo_params.terminating_mean_error = parseFloat($("#terminating_mean_error").val());
	request_data.algo_params.alpha_w = parseFloat($("#alpha_w").val());
	request_data.algo_params.alpha_e_avg_t = parseFloat($("#alpha_e_avg_t").val());
	request_data.algo_params.alpha_n0 = parseInt($("#alpha_n0").val());
	request_data.algo_params.max_iter = parseInt($("#max_iter").val());
	request_data.cluster_number = parseInt($("#cluster_number").val());
	request_data.beta_pickle_file_path = $("#beta_pickle_file_path").val();
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
	localStorage['fuzzy_c_task_id'] = data.task_id;
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
	localStorage['fuzzy_c_pid'] = data.pid;
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
	$("#beta_pickle_file_path").val(localStorage['beta_pickle_file_path'] || '')
	$('#pid').text(localStorage['fuzzy_c_pid'] || '');
	$('#task_id').text(localStorage['fuzzy_c_task_id'] || '');
}





