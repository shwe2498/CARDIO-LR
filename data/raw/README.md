# Raw Data Files

This directory should contain the raw data files needed for CARDIO-LR. These files are excluded from version control due to their size.

## Required Files

### BioASQ Dataset
- `training13b.json` - BioASQ challenge dataset

### MedQuAD Dataset
- `medquad.csv` - Medical Question-Answer Dataset

### SNOMED CT Files (January 2025 Release)
- `sct2_Concept_Snapshot_INT_20250601.txt`
- `sct2_Description_Snapshot-en_INT_20250601.txt`
- `sct2_Relationship_Snapshot_INT_20250601.txt`

### UMLS Files
- `MRCONSO.RRF` - Concept names and sources
- `MRREL.RRF` - Relationships
- `MRSTY.RRF` - Semantic types

## Getting the Data

### UMLS Data
UMLS data requires a license from the National Library of Medicine. Team members can get access at:
https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html

### SNOMED CT
SNOMED CT requires a license through the UMLS or directly from SNOMED International:
https://www.snomed.org/snomed-ct/get-snomed

### MedQuAD and BioASQ
These datasets are publicly available:
- MedQuAD: https://github.com/abachaa/MedQuAD
- BioASQ: http://participants-area.bioasq.org/datasets/