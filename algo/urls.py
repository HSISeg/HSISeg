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
    url(r'^pdhg_linear_ui$', views.pdhg_linear_ui, name='pdhg_linear_ui'),
    url(r'^initialization$', views.initialization, name='initialization'),
    url(r'^image_to_pickle$', views.image_to_pickle, name='image_to_pickle'),
    url(r'^weight_calculation$', views.weight_calculation, name='weight_calculation'),
    url(r'^quad_pdhg_ui$', views.quad_pdhg_ui, name='quad_pdhg_ui'),
    url(r'^beta_calculation$', views.beta_calculation, name='beta_calculation'),
    url(r'^fuzzy_c_ui$', views.fuzzy_c_ui, name='fuzzy_c_ui'),
    url(r'^get-image-to-pickle$', views.get_image_to_pickle, name='get-image-to-pickle'),
    url(r'', views.home, name='home'),
]
