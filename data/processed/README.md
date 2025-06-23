# Processed Data

This directory contains processed data files that are generated from the raw data. These files are excluded from version control due to their size but can be regenerated using the scripts in the project.

## Directory Structure

- `embeddings/` - Vector embeddings for medical terms and documents
- `kg/` - Knowledge graph files in processed format
- `models/` - Trained model files

## Regenerating the Files

After cloning the repository and obtaining the raw data files, you can regenerate these processed files using the following steps:

### 1. Knowledge Graph Generation

```bash
cd kg_construction
python umls_processor.py --filter-semantic-types
python snomed_processor.py --extract-cardio
python drugbank_processor.py
python knowledge_integrator.py
```

This will create the following files in `data/processed/kg/`:
- `umls_cardio_graph.pkl`
- `snomed_graph_2025.pkl`
- `integrated_cardio_graph.pkl`

### 2. Embedding Generation

```bash
cd retrieval
python embedding_generator.py
```

This will create embeddings in `data/processed/embeddings/`.

### 3. Model Training

```bash
cd gnn
python train_rgcn.py
```

This will create model files in `data/processed/models/`.

## Using Pre-processed Files on Jetstream

For team members using Jetstream, pre-processed files can be accessed at:
`/projects/cardio-lr/processed_data/`

Copy these files to your local `data/processed/` directory to avoid lengthy regeneration times.