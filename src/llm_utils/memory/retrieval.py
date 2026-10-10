# Source: https://github.com/mbchang/data-driven-characters

from typing import Any, ClassVar, List, Dict
from langchain.memory import VectorStoreRetrieverMemory

from langchain_core.documents import Document


class ConversationVectorStoreRetrieverMemory(VectorStoreRetrieverMemory):
    input_prefix: ClassVar[str] = "Human"
    output_prefix: ClassVar[str] = "AI"
    blacklist: ClassVar[List[str]] = []  # keys to ignore

    def _form_documents(
        self, inputs: Dict[str, Any], outputs: Dict[str, str]
    ) -> List[Document]:
        """Format context from this conversation to buffer."""
        # Each document should only include the current turn, not the chat history
        filtered_inputs = {
            k: v
            for k, v in inputs.items()
            if k != self.memory_key and k not in self.blacklist
        }
        texts = []
        for k, v in list(filtered_inputs.items()) + list(outputs.items()):
            if k == "input":
                k = self.input_prefix
            elif k == "response":
                k = self.output_prefix
            texts.append(f"{k}: {v}")
        page_content = "\n".join(texts)
        return [Document(page_content=page_content)]
