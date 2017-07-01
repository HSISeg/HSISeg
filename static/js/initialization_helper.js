$('#run_algorithm').on('click', function (e) {
	update_progress_bar("0",false);
	var request_data_json = construct_api_request_object();
	var cache_data = new Object();
	cache_data.task_id_cache = "init_task_id";
	cache_data.pid_cache = "init_pid";
	cache_data.output_file_path = "centroid_pickle_file_path";
	cache_data.fake_percentage = true;
	start_algorithm("/get-initial-centroid",request_data_json,cache_data);
})

function construct_api_request_object() {
	var request_data = new Object();
	request_data.image_pickle_file_path = $("#image_pickle_file_path").val();
	request_data.algo = $("#algo").val();
	request_data.output_path = $("#output_path").val();
	request_data.cluster_number = parseInt($("#cluster_number").val());
	request_data_json = JSON.stringify(request_data);

	append_to_log("Request JSON generated\n" + request_data_json);
	update_cache_input_variables();
  	return request_data_json;
}


function init() {
	$("#image_pickle_file_path").val(localStorage['image_pickle_file_path'] || '');
	$("#output_path").val(localStorage['output_path'] || '');
	$("#cluster_number").val(localStorage['cluster_number'] || '');
	$('#pid').text(localStorage['init_pid'] || '');
	$('#task_id').text(localStorage['init_task_id'] || '');

}