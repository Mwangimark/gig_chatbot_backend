from django.db import models
import uuid

class ChatLog(models.Model):
    message = models.TextField()
    response = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    session_id = models.UUIDField(default=uuid.uuid4, editable=False)

    def __str__(self):
        return f"Chat @ {self.timestamp}"
