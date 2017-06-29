from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^get-algo-params$', views.get_algo_params, name='get-algo-params'),
    url(r'^run-algo$', views.run_algo ,name='run_algo'),
    url(r'^get-algo-names$', views.get_algo_names, name='get_algo_names'),
    url(r'^get-beta$', views.get_beta, name='get_beta'),
    url(r'^get-weight$', views.get_weight, name='get_weight'),
    url(r'^kill-task$', views.kill_task, name='kill_task'),
    url(r'^get-task-status$', views.get_task_status, name='get_task_status'),
    url(r'^get-initial-centroid$', views.get_initial_centroid, name='get_initial_centroid'),
    url(r'', views.home, name='home'),
]
