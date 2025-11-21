from prometheus_client import Counter, Histogram

# Total requests per endpoint
RAG_REQUESTS = Counter(
    "rag_requests_total",
    "Total number of RAG API requests",
    ["endpoint"]
)

# Errors per endpoint
RAG_ERRORS = Counter(
    "rag_request_errors_total",
    "Total number of failed RAG API requests",
    ["endpoint"]
)

# Latency per endpoint
RAG_LATENCY = Histogram(
    "rag_request_latency_seconds",
    "Latency of RAG API requests in seconds",
    ["endpoint"],
)

# How many chunks we retrieved for each question
RAG_RETRIEVED_CHUNKS = Histogram(
    "rag_retrieved_chunks",
    "Number of retrieved chunks per request",
)