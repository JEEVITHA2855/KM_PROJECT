const SEVERITIES = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'];

function toUpperString(value) {
  if (value === null || value === undefined) return '';
  return String(value).trim().toUpperCase();
}

function normalizeConfidence(confidenceRaw) {
  if (confidenceRaw === null || confidenceRaw === undefined || confidenceRaw === '') return null;
  const numberValue = Number(confidenceRaw);
  if (Number.isNaN(numberValue)) return null;

  // Accept either 0..1 or 0..100
  if (numberValue > 1) return Math.max(0, Math.min(1, numberValue / 100));
  return Math.max(0, Math.min(1, numberValue));
}

function normalizeKeywords(keywordsRaw) {
  if (!keywordsRaw) return [];
  if (Array.isArray(keywordsRaw)) {
    return keywordsRaw
      .map((k) => String(k).trim())
      .filter(Boolean)
      .slice(0, 20);
  }

  return String(keywordsRaw)
    .split(/[,;\n]/g)
    .map((k) => k.trim())
    .filter(Boolean)
    .slice(0, 20);
}

function normalizeSimilarityList(rawList) {
  if (!rawList) return [];
  if (!Array.isArray(rawList)) return [];

  return rawList
    .map((item) => {
      if (!item) return null;
      const text = item.text ?? item.alert_text ?? item.alertText ?? item.content;
      const similarityRaw = item.similarity ?? item.score ?? item.similarity_score;
      const similarity = normalizeConfidence(similarityRaw);

      if (!text) return null;
      return {
        text: String(text).trim(),
        similarity: similarity ?? 0,
      };
    })
    .filter(Boolean)
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, 10);
}

export function normalizeAnalysisResponse(apiData, sourceText) {
  const severity = toUpperString(apiData?.severity ?? apiData?.severity_level ?? apiData?.severityLevel);
  const normalizedSeverity = SEVERITIES.includes(severity) ? severity : 'LOW';

  const department = toUpperString(apiData?.department ?? apiData?.dept ?? apiData?.team);

  const confidence = normalizeConfidence(apiData?.confidence ?? apiData?.confidence_score ?? apiData?.confidenceScore);

  const keywords = normalizeKeywords(apiData?.keywords ?? apiData?.extracted_keywords ?? apiData?.extractedKeywords);

  const semanticSimilar = normalizeSimilarityList(
    apiData?.semantic_similarity_results ?? apiData?.similarity_results ?? apiData?.similar_alerts ?? apiData?.semanticSimilar
  );

  const immediateActionRequired =
    apiData?.immediate_action_required ??
    apiData?.immediateActionRequired ??
    (normalizedSeverity === 'CRITICAL' || normalizedSeverity === 'HIGH');

  const immediateActionMessage =
    apiData?.immediate_action ?? apiData?.immediateAction ?? apiData?.action ?? apiData?.recommended_action ?? '';

  return {
    sourceText: sourceText ?? apiData?.text ?? '',
    severity: normalizedSeverity,
    department: department || 'UNASSIGNED',
    confidence,
    keywords,
    immediateAction: {
      required: Boolean(immediateActionRequired),
      message: String(immediateActionMessage || '').trim(),
    },
    semanticSimilar,
    raw: apiData,
  };
}

export function formatTimestamp(date) {
  try {
    return new Intl.DateTimeFormat(undefined, {
      year: 'numeric',
      month: 'short',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    }).format(date);
  } catch {
    return date.toISOString();
  }
}
