# KMRL Alert Detection Data Labeling Guidelines

## Objective
Label each KMRL document or paragraph with:
- Severity: Critical, High, Medium, Low
- Department: Finance, Safety, Operations, HR, etc.

## Steps
1. Read the document/paragraph.
2. Assign a severity label based on urgency and impact:
   - **Critical**: Immediate action required, major risk or incident.
   - **High**: Significant issue, needs prompt attention.
   - **Medium**: Moderate issue, monitor or address soon.
   - **Low**: Minor issue, routine or informational.
3. Assign the relevant department (choose from list or specify new if needed).

## Example Labeling Template (CSV)
| document_id | paragraph_id | text | severity | department |
|-------------|--------------|------|----------|------------|
| 001         | 1            | ...  | High     | Safety     |
| 001         | 2            | ...  | Low      | HR         |

## Notes
- If unsure, discuss with a domain expert.
- Use consistent criteria for all documents.
- Save labels in CSV or Excel format for model training.
