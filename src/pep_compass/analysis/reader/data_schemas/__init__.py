"""Versioned adapters for tracker output tables."""

from pep_compass.analysis.reader.data_schemas.base import DataSchema
from pep_compass.analysis.reader.data_schemas.registry import SCHEMAS, get_schema

__all__ = ["DataSchema", "SCHEMAS", "get_schema"]
