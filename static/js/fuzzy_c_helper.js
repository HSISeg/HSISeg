$('#run_algorithm').on('click', function (e) {
	update_progress_bar("0",false);
	var request_data_json = construct_api_request_object();
	var cache_data = new Object();
	cache_data.task_id_cache = "fuzzy_c_task_id";
	cache_data.pid_cache = "fuzzy_c_pid";
	cache_data.output_file_path = "";
	cache_data.fake_percentage = false;
	start_algorithm("/run-algo",request_data_json,cache_data);
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

function init() {
	$("#image_pickle_file_path").val(localStorage['image_pickle_file_path'] || '');
	$("#output_path").val(localStorage['output_path'] || '');
	$("#cluster_number").val(localStorage['cluster_number'] || '');
	$("#centroid_pickle_file_path").val(localStorage['centroid_pickle_file_path'] || '');
	$("#beta_pickle_file_path").val(localStorage['beta_pickle_file_path'] || '')
	$('#pid').text(localStorage['fuzzy_c_pid'] || '');
	$('#task_id').text(localStorage['fuzzy_c_task_id'] || '');
}





