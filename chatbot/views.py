# chatbot/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework import status
from langchain_core.messages import HumanMessage
import os

from .langgraph_flow import gig_chatbot_graph  # <-- the HF version above

class GiGChatView(APIView):
    permission_classes = [AllowAny]


    def post(self, request):

        user_message = request.data.get("message")
        if not user_message:
            return Response({"error": "No message provided."}, status=status.HTTP_400_BAD_REQUEST)

        if not request.session.session_key:
            request.session.save()
        thread_id = request.session.session_key

        state = {"messages": [HumanMessage(content=user_message)]}

        result = gig_chatbot_graph.invoke(
            state,
            config={"configurable": {"thread_id": thread_id}}
        )

        reply = result["messages"][-1].content if result.get("messages") else ""
        return Response({"reply": reply}, status=status.HTTP_200_OK)
