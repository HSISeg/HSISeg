$('#kill_task').on('click', function (e) {
	var task_id = parseInt($("#task_id").text());
	var request_data = new Object();
	request_data.task_id = task_id;
	$.post("/kill-task",JSON.stringify(request_data),kill_task_callback,"text");
	return;
})

$('#clear_logs').on('click', function (e) {
	$('#api_log').val("");
	return;
})

function kill_task_callback(data,status) {
	append_to_log("Response from Kill Task");
	append_to_log(data);
	return
}

function append_to_log(append_text) {
	var $api_log = $('#api_log');
	$api_log.val($api_log.val() + append_text + "\n********************\n")
	$api_log.scrollTop($api_log[0].scrollHeight);
	return
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

function update_cache_input_variables(){
	localStorage['image_pickle_file_path'] = $("#image_pickle_file_path").val();
	localStorage['output_path'] = $("#output_path").val();
	return;
}

function start_algorithm(url,request_data_json,cache_data) {
	var start_algorithm_callback_wrapper = function(cache_data) {
		return function (data,status) {
			start_algorithm_callback(data,status,cache_data);
		}
	}
	$.post(url,request_data_json,start_algorithm_callback_wrapper(cache_data),"text");
	return;
}

function start_algorithm_callback(data,status,cache_data) {
	var task_id_cache = cache_data.task_id_cache;
	append_to_log("Response For Algorithm Call")
	append_to_log("[" + status + "] " + data);
	var data = JSON.parse(data);
	localStorage[task_id_cache] = data.task_id;
	$('#task_id').text(data.task_id);
	poll_for_task(data.task_id,cache_data);
	return;
}

function poll_for_task(task_id,cache_data) {
	var poll_request = new Object();
	poll_request.task_id = task_id;

	var poll_for_task_callback_wrapper = function(cache_data) {
		return function(data,status) {
			poll_for_task_callback(data,status,cache_data);
		}
	}

	$.post("/get-task-status", JSON.stringify(poll_request), poll_for_task_callback_wrapper(cache_data), "text");
	return;
}


function poll_for_task_callback(data,status,cache_data) {
	var pid_cache = cache_data.pid_cache;
	var output_file_path = cache_data.output_file_path;
	append_to_log("Result of Task Poll")
	append_to_log("[" + status + "] " + data);
	var data = JSON.parse(data);
	$('#pid').text(data.pid);
	localStorage[pid_cache] = data.pid;
	update_progress_bar(data.percentage_done,cache_data.fake_percentage);
	if (!data.is_done){
		setTimeout(function(){ poll_for_task(data.task_id,cache_data); }, 2000);
	}else{
		if (output_file_path != ""){
			$("#" + output_file_path).val(data.result_file_name);
			localStorage[output_file_path] = data.result_file_name;
		}
		update_progress_bar(100,false);
	}
	return;
}