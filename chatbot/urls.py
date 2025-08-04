from django.urls import path
from .views import GiGChatView

urlpatterns = [
    path('chatbot/', GiGChatView.as_view(), name='gig-chat'),
]
