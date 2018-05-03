$('#run_algorithm').on('click', function (e) {
	update_progress_bar("0",false);
	var request_data_json = construct_api_request_object();
	var cache_data = new Object();
	cache_data.task_id_cache = "pdhg_linear_task_id";
	cache_data.pid_cache = "pdhg_linear_pid";
	cache_data.output_file_path = "";
	cache_data.fake_percentage = false;
	start_algorithm("/run-algo",request_data_json,cache_data);
})

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


function init() {
	$("#image_pickle_file_path").val(localStorage['image_pickle_file_path'] || '');
	$("#output_path").val(localStorage['output_path'] || '');
	$("#cluster_number").val(localStorage['cluster_number'] || '');
	$("#centroid_pickle_file_path").val(localStorage['centroid_pickle_file_path'] || '');
	$("#weight_pickle_file_path").val(localStorage['weight_pickle_file_path'] || '')
	$('#pid').text(localStorage['pdhg_linear_pid'] || '');
	$('#task_id').text(localStorage['pdhg_linear_task_id'] || '');
}