from config import KRIKRI_BASE_URL, KRIKRI_API_KEY
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from smolagents import Tool, LiteLLMModel, CodeAgent

class RetrieverTool(Tool):
    name = "retriever"
    description = "Χρησιμοποιεί τη σημασιολογική αναζήτηση με το ChromaDB για να ανακτήσει τις πιο σχετικές ενότητες εγγράφων."
    inputs = {
        "query": {
            "type": "string",
            "description": "Η ερώτηση προς απάντηση. Αυτή θα πρέπει να είναι σημασιολογικά κοντά στα έγγραφά σας.",
        }
    }
    output_type = "string"

    def __init__(self, persist_directory="chroma_langchain_db", **kwargs):
        super().__init__(**kwargs)

        # Load the embedding model
        self.embeddings_model = HuggingFaceEmbeddings(model_name="HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5")

        # Load the existing ChromaDB store
        self.vector_store = Chroma(
            collection_name="innoHub_store",
            embedding_function=self.embeddings_model,
            persist_directory=persist_directory
        )

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Η ερώτησή σου πρέπει να είναι string"

        # Retrieve relevant documents from ChromaDB
        docs = self.vector_store.similarity_search(query, k=1)

        # Format the retrieved documents for display
        return "\nΈγγραφα που ανέκτησα:\n" + "".join(
            [
                f"\n\n===== Έγγραφο {i+1} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )

# Initialize RetrieverTool with the existing ChromaDB store
retriever_tool = RetrieverTool()

krikri = LiteLLMModel(
    model_id="hosted_vllm/krikri-dpo",
    api_key=KRIKRI_API_KEY,
    api_base=KRIKRI_BASE_URL,
    temperature=0.1
)

agent = CodeAgent(
    tools=[retriever_tool], model=krikri, max_steps=4, verbosity_level=2
)

agent_output = agent.run("Πότε γίνεται το Σεμινάριο Εισαγωγή στα Μεγάλα Γλωσσικά Μοντέλα;")

print("Απάντηση:")
print(agent_output)