"""Acquire every original JSON page from the current SABIO-RK Export API."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode

from proteolysis_data_manager.acquisition.helpers.raw import acquire_files, fetch_bytes
from proteolysis_data_manager.utils.logger import get_custom_logger
from proteolysis_data_manager.utils.storage import read_download_receipt

logger = get_custom_logger(__name__)
API_URL = 'https://sabiork.h-its.org/export-api/sabio/kinlaw-entry/json'
DEFAULT_QUERY = 'ECNumber:3.4.*'
DEFAULT_PAGE_SIZE = 1000
VERSION = 'export-api/1.0'


def page_url(query, page, page_size):
    """Build a documented Export API request for one original page.

    :param str query: Explicit SABIO search expression.
    :param int page: One-based page number.
    :param int page_size: Requested number of source records per page.
    :return: Fully qualified official API URL.
    :rtype: str
    """
    return API_URL + '?' + urlencode({'q': query, 'page': page, 'pageSize': page_size})


def read_page(content):
    """Validate the API envelope while retaining original record contents.

    :param bytes content: Original response bytes.
    :return: Original decoded API response, including metadata and records.
    :rtype: dict
    :raises ValueError: If a page is malformed, truncated or an HTML response.
    """
    if not content.lstrip().startswith(b'{'):
        raise ValueError('SABIO-RK returned a non-JSON response; HTML is not kinetic data')
    payload = json.loads(content)
    if not isinstance(payload, dict) or not isinstance(payload.get('meta'), dict) or not isinstance(payload.get('data'), list):
        raise ValueError('Expected the SABIO-RK Export API meta/data envelope')
    meta = payload['meta']
    for key in ('page', 'page_size', 'total_count', 'total_pages'):
        if type(meta.get(key)) is not int:
            raise ValueError(f'Invalid SABIO-RK pagination field: {key}')
    if meta['page'] < 1 or meta['page_size'] < 1 or meta['total_count'] < 0:
        raise ValueError('Invalid SABIO-RK pagination values')
    expected_pages = (meta['total_count'] + meta['page_size'] - 1) // meta['page_size']
    if meta['total_pages'] != expected_pages or meta['page'] > max(1, expected_pages):
        raise ValueError('Inconsistent SABIO-RK total page count')
    expected_rows = min(meta['page_size'], max(0, meta['total_count'] - (meta['page'] - 1) * meta['page_size']))
    if len(payload['data']) != expected_rows:
        raise ValueError('SABIO-RK page contains fewer or more records than its metadata declares')
    if any(not isinstance(record, dict) or type(record.get('id')) is not int for record in payload['data']):
        raise ValueError('SABIO-RK source records must retain integer entry IDs')
    return payload


def validate(files):
    """Verify all pages, stable totals and unique entry identities.

    :param files: Source page filenames mapped to local paths.
    :raises ValueError: If pages are missing, inconsistent or overlapping.
    """
    first = read_page(Path(files['page_000001.json']).read_bytes())
    reference = first['meta']
    expected = {f'page_{page:06d}.json' for page in range(1, max(1, reference['total_pages']) + 1)}
    if set(files) != expected:
        raise ValueError('The SABIO-RK snapshot is missing expected pages')
    identifiers = set()
    for page, name in enumerate(sorted(files), 1):
        payload = first if page == 1 else read_page(Path(files[name]).read_bytes())
        meta = payload['meta']
        if meta['page'] != page or any(meta[key] != reference[key] for key in ('page_size', 'total_count', 'total_pages')):
            raise ValueError('SABIO-RK source pagination changed while acquiring the snapshot')
        for record in payload['data']:
            if record['id'] in identifiers:
                raise ValueError('SABIO-RK pages contain duplicate source entry IDs')
            identifiers.add(record['id'])
    if len(identifiers) != reference['total_count']:
        raise ValueError('SABIO-RK snapshot does not cover the declared source total')


def acquire(directory: Path, *, force=False, local_files=None, provenance=None,
            query=DEFAULT_QUERY, page_size=DEFAULT_PAGE_SIZE):
    """Store complete original JSON pages, without unit or field normalization.

    :param pathlib.Path directory: Destination ``raw/sabio`` directory.
    :param bool force: Retrieve a new snapshot instead of verifying the old one.
    :param local_files: Optional complete collection of original API page files.
    :param provenance: Original per-file receipts for local imports.
    :param str query: Explicit source query; defaults to all peptidase EC numbers.
    :param int page_size: Requested page size, between 1 and 1000.
    :return: Source receipt in the shared raw manifest.
    :rtype: dict
    :raises ValueError: If page validation, completeness or provenance checks fail.
    :raises OSError: If the official source is unavailable.
    """
    if not query.strip() or type(page_size) is not int or not 1 <= page_size <= 1000:
        raise ValueError('Provide a nonempty SABIO query and a page size between 1 and 1000')
    directory = Path(directory)
    previous = read_download_receipt(directory.parent.parent, 'sabio', missing_ok=True)
    payloads = {}
    first_name = 'page_000001.json'

    # Discover the source total once; reuse registered raw pages on ordinary reruns.
    if previous.get('files') and not force:
        first = read_page((directory / first_name).read_bytes())
    elif local_files:
        first = read_page(Path(local_files[first_name]).read_bytes())
    else:
        payloads[first_name] = fetch_bytes(page_url(query, 1, page_size))
        first = read_page(payloads[first_name])
    files = {f'page_{page:06d}.json': page_url(query, page, page_size)
             for page in range(1, max(1, first['meta']['total_pages']) + 1)}
    logger.info('SABIO-RK query %s: %s records in %s pages', query,
                first['meta']['total_count'], len(files))
    return acquire_files(directory, source='sabio', files=files, version=VERSION,
                         validate=validate, force=force, local_files=local_files,
                         provenance=provenance, payloads=payloads)
