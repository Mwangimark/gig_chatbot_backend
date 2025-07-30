from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny 

from .langgraph_flow import gig_chatbot_graph
from langchain_core.messages import HumanMessage

class GiGChatView(APIView):
    permission_classes = [AllowAny] 

    def post(self, request):
        user_message = request.data.get("message")
        if not user_message:
            return Response({"error": "No message provided."}, status=400)

        state = {
            "messages": [HumanMessage(content=user_message)]
        }

        result = gig_chatbot_graph.invoke(state)
        reply = result["messages"][-1].content
        return Response({"reply": reply})
    

