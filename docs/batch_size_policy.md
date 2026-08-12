\# Batch Size Policy



\## API-006 — Oversized Batch Behaviour



\### MVP Decision



Oversized-batch chunking or automatic batch splitting is \*\*out of MVP scope\*\*.



The current `/evaluate` implementation accepts a validated batch of submissions and forwards the batch to the existing Member 4 evaluation engine as a single evaluation request.



\### Current Behaviour



\- The API does not automatically split a large batch into smaller batches.

\- The API does not currently enforce a separate maximum batch-size limit.

\- The validated submission list is passed to the evaluation engine as one batch.

\- Batch-size limits may be introduced in a future version based on measured system performance and production requirements.



\### Rationale



The current MVP focuses on the core evaluation flow:



`Request → Validation → Evaluation Engine → Structured Response`



Introducing automatic chunking at this stage could change evaluation behaviour and add unnecessary complexity before practical batch-size and performance limits have been measured.



\### Future Consideration



A future implementation may introduce either:



1\. A maximum batch-size limit with a clear validation error, or

2\. Automatic batch chunking with controlled aggregation of results.



The decision should be based on measured performance, token usage, latency, and downstream integration requirements.



\### Status



\*\*API-006: Explicitly marked out of MVP scope.\*\*

