"""Project duration classes; censoring is never turned into an exact point."""
import json
import math
import pandas as pd

CLASS_BINS = [('short', 0.0, 1800.0), ('medium', 1800.0, 21600.0), ('long', 21600.0, math.inf)]
CLASS_SCHEME = 'project_v2; short<30min; medium=[30min,6h]; long>6h'


def point_class(point):
    if pd.isna(point) or point <= 0:
        return None
    return 'short' if point < 1800 else ('medium' if point <= 21600 else 'long')


def class_candidates(record):
    relation, point = record.get('relation'), record.get('half_life_seconds')
    if relation == 'eq' and pd.notna(point) and point > 0:
        return [point_class(point)]
    lower, upper = record.get('lower_bound_seconds'), record.get('upper_bound_seconds')
    if relation in {'lt', 'le'} and pd.notna(upper):
        return (['short'] if upper < 1800 or (upper == 1800 and relation == 'lt') else
                ['short', 'medium'] if upper <= 21600 else ['short', 'medium', 'long'])
    if relation in {'gt', 'ge'} and pd.notna(lower):
        return (['long'] if lower > 21600 or (lower == 21600 and relation == 'gt') else
                ['medium', 'long'] if lower >= 1800 else ['short', 'medium', 'long'])
    if relation == 'interval' and pd.notna(lower) and pd.notna(upper):
        return [name for name, intersects in [
            ('short', lower < 1800), ('medium', upper >= 1800 and lower <= 21600), ('long', upper > 21600)
        ] if intersects]
    return []


def add_duration_classes(frame):
    result = frame.copy()
    candidates = result.apply(class_candidates, axis=1)
    result['duration_class'] = candidates.map(lambda x: x[0] if len(x) == 1 else None)
    result['duration_class_candidates'] = candidates.map(json.dumps)
    result['duration_class_status'] = candidates.map(lambda x: 'determined' if len(x) == 1 else ('ambiguous' if x else 'unavailable'))
    # Reviewed project convention. Paper supports >1 h in vivo, not a 6 h
    # upper bound; this heuristic is documented in remarks_peplife2.md.
    override = (result.source.eq('peplife2') & result.source_record_id.eq('2686') &
                result.relation.eq('gt') & result.lower_bound_seconds.eq(3600) &
                result.publication_id.eq('25025626'))
    result.loc[override, 'duration_class'] = 'medium'
    result.loc[override, 'duration_class_status'] = 'project_heuristic_reviewed'
    result['duration_class_scheme'] = CLASS_SCHEME
    result['regression_label_available'] = result.measurement_kind.eq('half_life') & result.relation.eq('eq') & result.half_life_seconds.gt(0)
    return result
