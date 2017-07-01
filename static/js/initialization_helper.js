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
	request_data.algo = $("#algo").val();
	request_data.output_path = $("#output_path").val();
	request_data.cluster_number = parseInt($("#cluster_number").val());
	request_data_json = JSON.stringify(request_data);


	append_to_log("Request JSON generated\n" + request_data_json);
	update_cache_input_variables();
  	return request_data_json;
}

function update_cache_input_variables(){
	localStorage['image_pickle_file_path'] = $("#image_pickle_file_path").val();
	localStorage['output_path'] = $("#output_path").val();
	localStorage['cluster_number'] = $("#cluster_number").val();
	return;
}

function start_algorithm(request_data_json) {
	$.post("/get-initial-centroid",request_data_json,start_algorithm_callback,"text");
	return;
}

function start_algorithm_callback(data,status) {
	append_to_log("Response For Algorithm Call")
	append_to_log("[" + status + "] " + data);
	var data = JSON.parse(data);
	localStorage['task_id'] = data.task_id;
	localStorage['init_task_id'] = data.task_id;
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
	localStorage['init_pid'] = data.pid;
	update_progress_bar(data.percentage_done,true);
	if (!data.is_done){
		setTimeout(function(){ poll_for_task(data.task_id); }, 2000);
	}else{
		$('#centroid_pickle_file_path').val(data.result_file_name);
		localStorage['centroid_pickle_file_path'] = data.result_file_name;
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
	$('#pid').text(localStorage['init_pid'] || '');
	$('#task_id').text(localStorage['init_task_id'] || '');

}