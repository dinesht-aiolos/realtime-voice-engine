class EventAdapter:

    @staticmethod
    def normalize(provider: str, event: dict):
        if provider == "deepgram":
            return EventAdapter._deepgram(event)
        elif provider == "azure":
            return EventAdapter._azure(event)
        return event

    @staticmethod
    def _deepgram(event):
        if event.get("type") == "ConversationText":
            return {
                "type": "TRANSCRIPT",
                "role": event.get("role"),
                "text": event.get("content")
            }
        return event

    @staticmethod
    def _azure(event):
        if event.get("type") == "ConversationText":
            return {
                "type": "TRANSCRIPT",
                "role": event.get("role"),
                "text": event.get("content")
            }
        return event