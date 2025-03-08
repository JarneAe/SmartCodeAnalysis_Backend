from retrieval import retrieve, RetrievalRequest

docs = retrieve(RetrievalRequest(codebase_id="7b910a73-7a93-4744-b827-9dc7e0150201", query="How are ships added?", n=3))
print(docs)