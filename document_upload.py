"""Document extraction and contract-aware feature preparation for student predictions."""
import io, json, re, unicodedata
from pathlib import Path

import pandas as pd
from fastapi import HTTPException, UploadFile

BASE_DIR = Path(__file__).resolve().parent
CONTRACT = json.loads(
    (BASE_DIR / "data_contract.json").read_text(encoding="utf-8")
)

DEFAULTS = json.loads(
    (BASE_DIR / "upload_defaults.json").read_text(encoding="utf-8")
)

from schemas import (
    CATEGORICAL_FEATURES,
    ORDINAL_FEATURES,
    NUMERICAL_FEATURES,
)

ADMISSION_FEATURES = (
    CATEGORICAL_FEATURES
    + ORDINAL_FEATURES
    + NUMERICAL_FEATURES[:6]
)

SEMESTER_FEATURES = (
    CATEGORICAL_FEATURES
    + ORDINAL_FEATURES
    + NUMERICAL_FEATURES
)


def _norm(value):
    value = unicodedata.normalize('NFKD', str(value)).encode('ascii', 'ignore').decode().lower()
    return re.sub(r'[^a-z0-9]+', '', value)




def _read_upload(filename: str, content: bytes) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    try:
        if suffix == '.csv':
            return pd.read_csv(io.BytesIO(content))
        if suffix in ('.xlsx', '.xls'):
            return pd.read_excel(io.BytesIO(content))
        if suffix == '.docx':
            from docx import Document
            doc = Document(io.BytesIO(content))
            rows = []
            for table in doc.tables:
                for row in table.rows:
                    rows.append([cell.text.strip() for cell in row.cells])
            if rows:
                return pd.DataFrame(rows[1:], columns=rows[0])
            paragraphs = [p.text.strip() for p in doc.paragraphs if ':' in p.text]
            pairs = [p.split(':', 1) for p in paragraphs]
            if pairs:
                return pd.DataFrame([{k.strip(): v.strip() for k, v in pairs}])
            raise ValueError('No table or Feature: Value fields found in DOCX.')
        if suffix == '.pdf':
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(content))
            text = '\n'.join(page.extract_text() or '' for page in reader.pages)
            pairs = re.findall(r'^\s*([^:\n]+):\s*([^\n]+)', text, flags=re.M)
            if pairs:
                return pd.DataFrame([{k.strip(): v.strip() for k, v in pairs}])
            raise ValueError('No extractable text in PDF. Scanned PDFs need OCR; try DOCX, CSV or Excel.')
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'Could not read file: {exc}') from exc
    raise HTTPException(status_code=400, detail='Supported formats: .xlsx, .xls, .csv, .docx, .pdf')


def prepare_preview(filename: str, content: bytes, model_name: str):
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail='File size must be 10 MB or less.')
    frame = _read_upload(filename, content)
    if frame.empty:
        raise HTTPException(status_code=400, detail='The uploaded file contains no student records.')
    features = ADMISSION_FEATURES if model_name == 'Model A' else SEMESTER_FEATURES
    source_columns = {_norm(col): col for col in frame.columns}
    matched, defaults, invalid, unmatched = {}, [], [], []
    source_row = frame.iloc[0].to_dict()  # First data row represents one student.
    for feature in features:
        source_col = source_columns.get(_norm(feature))
        if source_col is None:
            matched[feature] = DEFAULTS[feature]
            defaults.append({'feature': feature, 'value': DEFAULTS[feature], 'reason': 'Column not found'})
            continue
        raw = source_row[source_col]
        if pd.isna(raw) or str(raw).strip() == '':
            matched[feature] = DEFAULTS[feature]
            defaults.append({'feature': feature, 'value': DEFAULTS[feature], 'reason': 'Blank value'})
            continue
        try:
            rule = CONTRACT[feature]
            value = float(raw)
            if not value.is_integer() and rule['type'] in ('categorical', 'ordinal'):
                raise ValueError('Expected an integer code')
            if rule['type'] in ('categorical', 'ordinal'):
                value = int(value)
                if value not in rule['allowed_values']:
                    raise ValueError(f'Allowed values: {rule["allowed_values"]}')
            else:
                if not float(rule['minimum']) <= value <= float(rule['maximum']):
                    raise ValueError(f'Allowed range: {rule["minimum"]} to {rule["maximum"]}')
            matched[feature] = value
        except (TypeError, ValueError):
            invalid.append({'feature': feature, 'source_column': str(source_col), 'value': str(raw), 'reason': 'Value does not satisfy the data contract'})
    used_cols = {c for c in source_columns.values() if _norm(c) in {_norm(f) for f in features}}
    unmatched = [str(c) for c in frame.columns if c not in used_cols]
    return {
        'model': model_name, 'row_used': 1, 'total_rows': int(len(frame)),
        'features': matched, 'defaults_applied': defaults,
        'invalid_values': invalid, 'unmatched_columns': unmatched,
        'can_predict': len(invalid) == 0,
        'message': 'Review extracted values and defaults before prediction.'
    }
