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
	$api_log.append(append_text + "\n********************\n");
	$api_log.scrollTop($api_log[0].scrollHeight);
	return
}