from django.urls import path
from .views import GiGChatView

urlpatterns = [
    path('chat/', GiGChatView.as_view(), name='gig-chat'),
]
