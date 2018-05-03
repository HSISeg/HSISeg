from django.conf.urls import include, url
from django.contrib import admin
from algo import urls as algo_urls

urlpatterns = [
    # Examples:
    # url(r'^$', 'HSISeg.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),

    url(r'^admin/', admin.site.urls),
    url(r'', include(algo_urls))

]
