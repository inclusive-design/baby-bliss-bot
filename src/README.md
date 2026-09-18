# Source Code Structure

The `src/` directory contains the core data processing and Bliss symbol processing components for the Baby Bliss Bot project.

## Overview

This directory is organized into two main components that work together:

1. **`process_data/`** - Data preparation pipeline
2. **`bliss_engine/`** - Core symbol processing engine
3. **`data/`** - Shared data resources

```
src/
├── process_data/          # Data preparation scripts
├── bliss_engine/          # Core Bliss symbol processing
├── data/                  # Shared data resources
└── requirements.txt       # Python dependencies
```

## Component Workflow

The components follow a sequential workflow for working with Blissymbolics data:

```
Raw Data Sources
        ↓
  process_data/
  (Prepare & Clean)
        ↓
  src/data/bliss_dict/
  (Processed Dictionary)
        ↓
  bliss_engine/
  (Symbol Analysis & Composition)
        ↓
  Applications & Models
```

## Process Data (`process_data/`)

**Purpose**: Prepare and clean raw Blissymbolics data before it can be used by the engine.

**Workflow**:

1. Expand the Blissymbolics dictionary with multilingual descriptions
2. Clean and standardize glosses
3. Identify duplicate glosses for data quality analysis

**Key Outputs**:

- `bliss_dict_multi_langs.json` - Clean, standardized multilingual dictionary
- `duplicate_glosses.json` - Quality assurance report

**For Details**: See [process_data/README.md](./process_data/README.md)
