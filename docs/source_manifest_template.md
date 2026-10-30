# Dataset Source Manifest Template

Use this manifest template to document external dataset provenance, collection terms, and privacy sanitization before ingesting data into the pipeline.

```yaml
# Dataset Provenance & Ingestion Manifest
dataset_name: "romanized-nepali-social-sentiment"
version: "1.0.0"
collection_date: "2026-10-29"
source_platform: "Reddit / Twitter / YouTube"
source_url: "https://example.com/dataset-source"
terms_of_service_url: "https://example.com/tos"
license: "CC-BY-4.0"

collection:
  method: "API Export / Scraping"
  query_or_subreddits:
    - "r/nepal"
  date_range_start: "2025-01-01"
  date_range_end: "2026-06-30"

privacy_and_sanitization:
  author_identifiers_hashed: true
  hash_salt_retained: false
  pii_scrubbed:
    - "@mentions"
    - "phone numbers"
    - "email addresses"
  fields_collected:
    - "id"
    - "text"
    - "created_utc"
    - "score"
  fields_removed:
    - "author"
    - "author_fullname"
    - "user_ip"

labeling:
  scheme: "3-class (positive, negative, neutral)"
  annotator_type: "human_expert / consensus / automated_heuristic"
  annotators_count: 2
  inter_annotator_agreement_score: 0.84

notes: |
  Ingested records undergo schema validation, deduplication, and
  leakage-safe reproducible splitting using `romanized-nepali dataset`.
```
