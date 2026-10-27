# Security notes

Identity is established through an expiring signed bearer token. Every SQL resource lookup combines the resource ID with the authenticated owner ID. Vector queries and deletion include owner ID; document IDs supplied by a client are checked against ready records owned by that user. Source metadata is derived from matching vectors.

The old prototype's username field and shared Chroma index remain in `rag_app/` for reference. The new API uses a separate `runtime/chroma` collection and does not import legacy data. Do not expose the legacy UI or index to multiple users.

Upload names are displayed but never used as filesystem paths. A server UUID is the storage name. File size and format are checked before indexing. PDF page count, DOCX uncompressed size, extracted text and chunk count are bounded. API errors avoid provider response bodies and credentials.

Remaining operational controls: TLS and reverse proxy limits for remote deployment, backups and retention policy, secret rotation, rate limiting, malware scanning, dependency scanning, logging and monitoring. The committed legacy PDF and JSON file may contain private information; content was deliberately not examined or migrated. Investigate ownership and sensitivity before deciding whether to purge historical commits, with separate explicit authorization. Git ignore does not remove tracked files.
