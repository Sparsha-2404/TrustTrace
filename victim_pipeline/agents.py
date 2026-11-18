import os
from typing import Optional

# Defer heavy imports / initialization at import-time so tests that merely
# import this module do not hang when external services are unavailable.
_chroma_client = None
_knowledge_base = None
_llm_provider = None
_chroma_init_error = None

def _init_chroma():
    """Lazily initialize the shared persistent Chroma store."""
    global _chroma_client, _knowledge_base, _chroma_init_error
    if _knowledge_base is not None and not isinstance(_knowledge_base, _InMemoryCollection):
        return
    try:
        from victim_pipeline.chroma_store import get_chroma_store
        _chroma_client, _knowledge_base = get_chroma_store()
        _chroma_init_error = None
    except Exception as exc:
        # The in-memory store exists only to keep unit tests/imports usable.
        # main.bootstrap_system explicitly rejects it for real runs.
        _chroma_client = None
        _knowledge_base = None
        _chroma_init_error = exc


def is_persistent_knowledge_base() -> bool:
    """Whether the active pipeline collection is backed by PersistentClient."""
    return _knowledge_base is not None and not isinstance(_knowledge_base, _InMemoryCollection)


def get_chroma_initialization_error():
    return _chroma_init_error


class _InMemoryCollection:
    """Minimal in-memory fallback implementing the small Chroma subset used.

    This allows tests and interactive runs to operate even when chromadb
    isn't installed.
    """
    def __init__(self):
        self._docs = []
        self._metadatas = []
        self._ids = []

    def add(self, documents, metadatas=None, ids=None):
        metadatas = metadatas or [None] * len(documents)
        ids = ids or [f"doc_{len(self._ids) + i}" for i in range(len(documents))]
        if len(documents) != len(metadatas) or len(documents) != len(ids):
            raise ValueError("documents, metadatas and ids must have equal lengths")
        if len(set(ids)) != len(ids) or any(existing in set(ids) for existing in self._ids):
            raise ValueError("duplicate document id")
        for doc, md, id_ in zip(documents, metadatas, ids):
            self._docs.append(doc)
            self._metadatas.append(md)
            self._ids.append(id_)

    def get(self):
        return {"ids": list(self._ids), "documents": list(self._docs), "metadatas": list(self._metadatas)}

    def delete(self, ids=None):
        if not ids:
            self._docs.clear(); self._metadatas.clear(); self._ids.clear(); return
        new_docs, new_mds, new_ids = [], [], []
        for doc, md, id_ in zip(self._docs, self._metadatas, self._ids):
            if id_ not in ids:
                new_docs.append(doc); new_mds.append(md); new_ids.append(id_)
        self._docs, self._metadatas, self._ids = new_docs, new_mds, new_ids

    def count(self):
        return len(self._ids)

    def query(self, query_texts=None, n_results=3):
        # Very simple: return the first n_results documents
        docs = self._docs[:n_results]
        mds = self._metadatas[:n_results]
        return {"documents": [docs], "metadatas": [mds]}


def _init_llm_provider():
    """Lazily initialize and cache the LLM provider."""
    global _llm_provider
    if _llm_provider is not None:
        return
    try:
        from llm.llm_provider import get_llm_provider

        _llm_provider = get_llm_provider()
    except Exception as exc:
        # Never silently substitute a deterministic mock when the configured
        # real provider cannot be initialised. That would contaminate demos and
        # research runs with outputs that look successful but are not model calls.
        raise RuntimeError(
            "Failed to initialise the configured LLM provider. Check config.yaml, "
            "the provider package, and the LM Studio/API endpoint. No mock fallback was used."
        ) from exc

# ── Agent definitions (simplified without CrewAI) ───────────────────────────────

AGENT_NAMES = ["Retriever", "Planner", "Executor", "Generator"]
AGENT_ORDER = ["Retriever", "Planner", "Executor", "Generator"]

# Agent function backed by the configured real LLM provider.
def _invoke_agent(agent_name: str, task: str, context: str = "") -> str:
    """Invoke a single agent using the configured LLM provider."""
    if task and isinstance(task, str) and task.startswith("[BLOCKED]"):
        return "I cannot process this request because it was identified as malicious."
    # Ensure the configured LLM is available; never fabricate a fallback response.
    _init_llm_provider()
    system_prompt = f"You are the {agent_name} agent."
    if context:
        task = f"Context: {context}\n\nTask: {task}"
    if _llm_provider is None:
        raise RuntimeError(
            "No LLM provider is available. Configure a real provider in config.yaml; "
            "the pipeline will not fabricate a mock response."
        )
    if os.environ.get("TRUSTTRACE_STRICT_EVAL", "").strip().lower() in {"1", "true", "yes"}:
        if _llm_provider.__class__.__name__ == "MockProvider":
            raise RuntimeError("Strict evaluation mode forbids MockProvider.")
    output = _llm_provider.generate(task, system_prompt=system_prompt)
    if os.environ.get("TRUSTTRACE_STRICT_EVAL", "").strip().lower() in {"1", "true", "yes"} and str(output).lstrip().startswith("[mock:"):
        raise RuntimeError("Strict evaluation mode detected mock output.")
    return output


def run_pipeline(query: str) -> dict:
    """
    Run the unmonitored four-agent pipeline.
    This low-level function is reserved for baseline/control experiments;
    interactive or defended execution must use run_pipeline_stepwise through
    TrustTraceRuntime.
    """
    outputs = {}
    
    # Step 1: Retriever
    context = retrieve_context(query)
    outputs["Retriever"] = str(context)
    
    # Step 2: Planner
    outputs["Planner"] = _invoke_agent("Planner", f"Create a step-by-step plan based on: {query}\nContext: {context}")
    
    # Step 3: Executor
    outputs["Executor"] = _invoke_agent("Executor", f"Execute this plan: {outputs['Planner']}")
    
    # Step 4: Generator
    outputs["Generator"] = _invoke_agent("Generator", f"Write the final answer combining: {outputs['Planner']} and {outputs['Executor']}")
    
    return outputs


def retrieve_context(query: str, top_k: int = 3, *, exclude_quarantined: bool = False) -> list:
    """Retrieve relevant documents, optionally excluding persisted quarantine.

    A0/control retrieval keeps the historical unfiltered behaviour. Defended
    per-hop runs request quarantine filtering explicitly. When filtering, query
    the full collection before truncating so quarantined top-ranked records do
    not hide eligible records further down the ranking.
    """
    _init_chroma()
    if _knowledge_base is None:
        raise RuntimeError(
            "Knowledge base is unavailable. Initialize persistent ChromaDB before retrieval."
        )
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")

    query_count = int(_knowledge_base.count()) if exclude_quarantined else top_k
    if query_count <= 0:
        return []
    results = _knowledge_base.query(query_texts=[query], n_results=max(1, query_count))
    documents = results.get('documents', [[]])[0] or []
    metadatas = results.get('metadatas', [[]])[0] or []
    pairs = list(zip(documents, metadatas))
    if exclude_quarantined:
        pairs = [
            (document, metadata or {})
            for document, metadata in pairs
            if not bool((metadata or {}).get("quarantined", False))
        ]
    return pairs[:top_k]


def run_pipeline_stepwise(
    query: str,
    on_agent_start=None,
    on_agent_complete=None,
    quarantined=None,
    block_generator=False,
    should_execute=None,
    exclude_quarantined: bool = False,
) -> dict:
    """
    Run the pipeline step-by-step, returning intermediate outputs.
    This is used by TrustTrace to monitor each agent individually.

    Args:
        query: The user query to process
        on_agent_start: Callback function called before each agent executes (agent, task, extra)
        on_agent_complete: Callback function called after each agent executes (agent, output, blocked)
        quarantined: Set of quarantined agent names to skip. This is a static
            snapshot taken before the run starts — kept for backward
            compatibility, but agents whose trust collapses *during* this
            same run (as a result of an earlier agent's output) will NOT be
            in this set yet. Use `should_execute` for that.
        block_generator: If True, block Generator execution and return safe refusal.
            Same staleness caveat as `quarantined` applies.
        should_execute: Optional callable(agent_name) -> bool, called live
            immediately before each agent step (after all prior agents in
            this run have already updated trust). This is what actually
            lets a mid-run trust collapse stop a downstream agent — a
            static `quarantined` set computed before the pipeline started
            cannot see trust drops caused by this same run's own agents.
    """
    outputs = {}
    quarantined = set(quarantined or set())

    def _allowed(agent: str) -> bool:
        if agent in quarantined:
            return False
        if should_execute is not None and not should_execute(agent):
            quarantined.add(agent)
            return False
        return True

    if query and isinstance(query, str) and query.startswith("[BLOCKED]"):
        return {
            "Retriever": "[BLOCKED] Request blocked before retrieval.",
            "Planner": "[BLOCKED] Request blocked before planning.",
            "Executor": "[BLOCKED] Request blocked before execution.",
            "Generator": "[BLOCKED] I cannot complete this request because it was identified as malicious before the model pipeline executed.",
        }
    
    # Step 1: Retriever
    if _allowed("Retriever"):
        if on_agent_start:
            on_agent_start("Retriever", query, None)
        context = retrieve_context(query, exclude_quarantined=exclude_quarantined)
        if on_agent_start:
            # A security callback may return a sanitized retrieval result. The
            # transformed value is used downstream; merely observing/rejecting
            # a document in a callback must never leave the original context in
            # the Planner prompt.
            reviewed_context = on_agent_start("KnowledgeBase", query, context)
            if reviewed_context is not None:
                if not isinstance(reviewed_context, (list, tuple)):
                    raise TypeError("KnowledgeBase security callback must return a list/tuple or None")
                context = list(reviewed_context)
        outputs["Retriever"] = str(context)
        if on_agent_complete:
            on_agent_complete("Retriever", outputs["Retriever"], False)
    else:
        context = ""
        outputs["Retriever"] = "[QUARANTINED]"
    
    # Step 2: Planner — re-checked live: Retriever's output may have just
    # dropped trust enough to stop Planner from ever seeing it.
    if _allowed("Planner"):
        planner_task = f"Create a step-by-step plan based on: {query}\nContext: {context}"
        if on_agent_start:
            on_agent_start("Planner", planner_task, None)
        outputs["Planner"] = _invoke_agent("Planner", planner_task)
        if on_agent_complete:
            on_agent_complete("Planner", outputs["Planner"], False)
    else:
        outputs["Planner"] = "[QUARANTINED]"
    
    # Step 3: Executor — re-checked live after Planner's trust update.
    if _allowed("Executor"):
        if on_agent_start:
            on_agent_start("Executor", outputs.get("Planner", ""), None)
        outputs["Executor"] = _invoke_agent("Executor", f"Execute this plan: {outputs['Planner']}")
        if on_agent_complete:
            on_agent_complete("Executor", outputs["Executor"], False)
    else:
        outputs["Executor"] = "[QUARANTINED]"
    
    # Step 4: Generator — re-checked live after Executor's trust update,
    # in addition to the upfront block_generator flag.
    if _allowed("Generator") and not block_generator:
        if on_agent_start:
            on_agent_start("Generator", outputs.get("Executor", ""), None)
        outputs["Generator"] = _invoke_agent("Generator", f"Write the final answer combining: {outputs['Planner']} and {outputs['Executor']}")
        if on_agent_complete:
            on_agent_complete("Generator", outputs["Generator"], False)
    elif "Generator" in quarantined:
        outputs["Generator"] = "[BLOCKED] I cannot complete this request because it was blocked by the compromised pipeline."
        if on_agent_complete:
            on_agent_complete("Generator", outputs["Generator"], True)
    else:
        outputs["Generator"] = "[BLOCKED] I cannot complete this request due to security concerns. The system detected potentially malicious content in the input."
        if on_agent_complete:
            on_agent_complete("Generator", outputs["Generator"], True)
    
    return outputs


def clear_attacker_documents(collection) -> int:
    """
    Clear any documents marked as malicious from the knowledge base.
    Returns the number of documents removed.
    """
    try:
        # Get all documents
        all_data = collection.get()
        if not all_data or not all_data.get('ids'):
            return 0
        
        # Filter out documents with malicious metadata
        ids_to_delete = []
        for i, metadata in enumerate(all_data.get('metadatas', [])):
            if metadata and metadata.get('malicious', False):
                ids_to_delete.append(all_data['ids'][i])
        
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            return len(ids_to_delete)
        
        return 0
    except Exception as e:
        print(f"Error clearing attacker documents: {e}")
        return 0


def reset_knowledge_base_for_benign(collection) -> None:
    """
    Reset the knowledge base to a clean state for benign runs.
    This removes all documents and re-initializes the collection.
    """
    # Clear the collection and immediately restore trusted baseline references.
    # Previously this function emptied the KB after it had been seeded, leaving
    # subsequent benign/indirect-injection runs with an empty retrieval corpus.
    all_data = collection.get()
    ids = (all_data or {}).get("ids", [])
    if ids:
        collection.delete(ids=ids)
    from memory.chroma_recovery import ensure_collection_ready
    ensure_collection_ready(collection)
    print("Knowledge base reset and trusted baseline references restored.")


def add_benign_documents(collection) -> None:
    """
    Add benign sample documents to the knowledge base for testing.
    """
    benign_docs = [
        "The capital of France is Paris, known for the Eiffel Tower.",
        "Python is a high-level programming language popular for data science.",
        "Machine learning is a subset of artificial intelligence.",
        "The Earth orbits around the Sun approximately every 365 days.",
        "Water boils at 100 degrees Celsius at sea level.",
    ]
    
    try:
        collection.add(
            documents=benign_docs,
            metadatas=[{"malicious": False} for _ in benign_docs],
            ids=[f"doc_{i}" for i in range(len(benign_docs))]
        )
        print(f"Added {len(benign_docs)} benign documents to knowledge base.")
    except Exception as e:
        print(f"Error adding benign documents: {e}")


# Ensure a top-level `knowledge_base` symbol exists for imports elsewhere.
# Initialize Chroma if available, otherwise expose the in-memory fallback.
_init_chroma()
if _knowledge_base is None:
    _knowledge_base = _InMemoryCollection()

# Public alias expected by other modules
knowledge_base = _knowledge_base
