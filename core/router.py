from providers.deepgram_provider import DeepgramProvider
from providers.azure_provider import AzureProvider

class ProviderRouter:

    def get_provider(self, name: str):
        if name == "deepgram":
            return DeepgramProvider()
        elif name == "azure":
            return AzureProvider()
        else:
            raise ValueError(f"Unsupported provider: {name}")