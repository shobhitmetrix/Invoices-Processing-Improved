#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parallel Image Processing Pipeline for P&P Rounds
- Downloads images from CSV links (parallel)
- Renames and moves files (parallel)
- Converts PDFs to JPEG in batches (parallel per PDF)
- Removes duplicate images (sequential – hash comparison)
"""

import os
import re
import gc
import csv
import json
import math
import random
import shutil
import argparse
import warnings
from datetime import datetime
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import imagehash
from PIL import Image
from pdf2image import convert_from_path
from pdf2image.pdf2image import pdfinfo_from_path

warnings.filterwarnings("ignore")

# ================================
# Configuration & Constants
# ================================
POPPLER_PATH = r"C:\Users\Admin\Downloads\P&P Streamline 1\P&P Streamline\Release-25.12.0-0\poppler-25.12.0\Library\bin"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COOKIES_FILE = os.path.join(SCRIPT_DIR, "cookies.txt")

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
)

# Fallback only if cookies.txt is missing (prefer pasting a fresh Cookie into cookies.txt).
DEFAULT_COOKIE = (
    "MicrosoftApplicationsTelemetryFirstLaunchTime=; "
    "MSFPC=GUID=cffca485bc334f2fa444cd17b002e053&HASH=cffc&LV=202504&V=4&LU=1744018727855; "
    "rldas=|||; rtFa=cCgtfQ45nvdUNT/tf+4oGaZDRFJpRapy+Jxj5jC+rzomYjk5OWJmNmItODA1NS00MzYwLWI4NTQtYTg0M2Y3ZGVjOGJlIzEzNDMzNDE5OTY1MDU2MTEwMCMwMTdiMzlhMi0yMDE5LTgwMDAtYWJmZS1iODllOGJhYTc5ZTUjc2hvYmhpdCU0MG1ldHJpeHJhLmNvbSMxOTY0NjgjZHh0akZxZFJGbEpfVm9wZUxtZEU1a1FpVkJNI2R4dGpGcWRSRmxKX1ZvcGVMbWRFNWtRaVZCTQyJ8oDAxqi0ZFUQklLP8Go/r5mWDA/lTEUscXigWO9p4SnOAOCuEbjcsO1IgkpqCRa89xy1sUEy9fx8ohsfdMH/E3U+oxPBDf0EW61S85tKGmM8iVfeVxco2sKnMkGJclT0/fAhlspTUJSCt0ytYpoNF3kSB6c9AuX607nTyBz1sTw2Anr8QQIYL7Hhfgs2YurHjJAv5aJh9cdn5ddCXx9QWmh7ZH9yqMULFAFLo80nfhIiXTB4aQmogH2lqlnxL+wa7BFTPMzj0yacWGgGQWwwyUyRfBkRmJkI1LaNNe6/eUDZp0YwSgw6EXDIjz1hjW5Mo74h9lzLKmzpSkhkDOrTAAAA; ExcelWacDataCenterSetTime=2026-03-12T15:05:14.233Z; WacDataCenterSetTime=2026-03-12T15:05:14.233Z; FeatureOverrides_experiments=[]; MicrosoftApplicationsTelemetryDeviceId=c7544eba-4e74-4288-a31a-a1fa0188de5a; ai_session=byj0rqtktT6XgbB0N6lQPb|1758184971250|1758187108441; "
    "FedAuth=77u/PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz48U1A+VjE1LDBoLmZ8bWVtYmVyc2hpcHwxMDAzMjAwNDc4YjA3ZWYwQGxpdmUuY29tLDAjLmZ8bWVtYmVyc2hpcHxzaG9iaGl0QG1ldHJpeHJhLmNvbSwxMzM4OTE2Nzc1ODAwMDAwMDAsMTMzODgwNDczNTYwMDAwMDAwLDEzNDM2Mjc1Mjk1NzA3ODY4NCwxMjUuMjIuMjM3LjUwLDMsYjk5OWJmNmItODA1NS00MzYwLWI4NTQtYTg0M2Y3ZGVjOGJlLCwwMDNmNWFkOS0wZTNiLTZiYmEtNjNhNS0wNzYxYmQzY2VmZjUsYTkwYjMyYTItYjA4Yi04MDAwLTdhMDktNTVlNTNiMDUzMmYwLDEyODI0MmEyLTkwOWQtODAwMC1iMTNjLWNmYzU2NjkyYzgxNywsMCwxMzQzNTkyOTY5NTY5NzkwODAsMTM0MzYxMDI0OTU2OTc5MDgwLCwsZXlKNGJYTmZZMk1pT2lKYlhDSkRVREZjSWwwaUxDSjRiWE5mYzNOdElqb2lNU0lzSW5CeVpXWmxjbkpsWkY5MWMyVnlibUZ0WlNJNkluTm9iMkpvYVhSQWJXVjBjbWw0Y21FdVkyOXRJaXdpZFhScElqb2lYMTlKT0RNM2JsVmpSWGxOTkhFdFpHb3pTbUpCUVNJc0ltRjFkR2hmZEdsdFpTSTZJakV6TXpnNU1UWTNOelU0TURBd01EQXdNQ0o5LDI2NTA0Njc3NDM5OTk5OTk5OTksMTM0MzE0MjQxNjUwMDAwMDAwLGUyZjBmNWQzLTlhMTktNGIyOC05MDJmLTZlNGEzZTJlMDY3ZCwsLCwsLDExNTI5MjE1MDQ2MDY4NDY5NzYsLDE5NjQ2OCwxeklXR0I1WDFkLXZNeGxTSzAtS21tMHZ1dVEsLDE5NjQ2OCwxeklXR0I1WDFkLXZNeGxTSzAtS21tMHZ1dVEsSis1TDE0N3N2aWtxbEdxU3NYb0U0d0JqVHNCL3l5bGVtYXQ3c2tIVkZmS2NuU3k4dHJuQXgrRVFkSGJBMEdZaW83M2FrRmsrT29LQmVWaVN3czRJMW1xRnkrUmVHT3hNZ3hSOVl3VUkxOTk3cy80VCt6MTZvc05FYWdabDdHbXlpM3gybS9lV2lmSUwveFBaTmowRW10bktza1pseGdQY0ZqYlJtUE9rQ3pHUTRjcTRHdFVqRWFhQ1lnc2w5anVZVVpBNklLT3pCakxPdHJVUzdrRGhPa0dpdnloZ0lmWjJONUdzb0dwWDA3T3NqYkZpcVExNjB5a3JpQkVOMk1ISHQ0aFhkS3NVem1kY3Nxek9BRGhqUkN2cExPelNvLzdVbG0zdGNXNVFJNGJ5SmtLSHRpZTdCZHdKYzh6U0pyVDRRNVduQlpTRXdESUZ2d2hIK25VOTN3PT08L1NQPg=="
)


SUPPORTED_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp',
    '.pdf', '.jfif', '.heic', '.heif', '.svg', '.raw', '.cr2'
}
NUM_WORKERS = 10   # Number of parallel threads
DOWNLOAD_HEADERS = {}


# ================================
# Helper Functions
# ================================
def build_download_headers():
    """Prefer cookies.txt (full browser Cookie header) over DEFAULT_COOKIE in script."""
    cookie = DEFAULT_COOKIE
    source = "DEFAULT_COOKIE in step1_download_invoice.py"
    if os.path.isfile(COOKIES_FILE):
        with open(COOKIES_FILE, encoding="utf-8") as f:
            raw = f.read().strip()
        if raw:
            cookie = raw.split(":", 1)[1].strip() if raw.lower().startswith("cookie:") else raw
            source = COOKIES_FILE
    return {"User-Agent": DEFAULT_USER_AGENT, "Cookie": cookie}, source


def parse_url_cell(value):
    if pd.isna(value):
        return []
    return [u.strip() for u in str(value).split(",") if u.strip().lower().startswith("http")]


def is_valid_file_content(data, content_type=""):
    if not data or len(data) < 4:
        return False
    ct = (content_type or "").lower()
    if "text/html" in ct:
        return False
    head = data[:500].lstrip().lower()
    if head.startswith(b"<!doctype") or head.startswith(b"<html") or head.startswith(b"<!-- copyright"):
        return False
    if data[:4] == b"%PDF" or data[:3] == b"\xff\xd8\xff" or data[:4] == b"\x89PNG":
        return True
    if data[:4] in (b"GIF8", b"RIFF"):
        return True
    if "image/" in ct or "application/pdf" in ct:
        return True
    return False


def file_name(url_list):
    """Extract filename from Uploads/ path."""
    fin_lst = []
    for i in list(url_list):
        parts = str(i).split('Uploads/')
        if len(parts) > 1:
            fin_lst.append(re.sub(r"^\d{2}-\d{4}/", "", parts[1]))
    return fin_lst


def download_image(url, output_dir, idx):
    """Download a single image (used in parallel)."""
    url = str(url).strip()
    if not url.lower().startswith("http"):
        return f"SKIP: not a URL ({url[:80]})"
    try:
        response = requests.get(url, headers=DOWNLOAD_HEADERS, timeout=30)
        if response.status_code != 200:
            return f"Failed {url} (status {response.status_code})"
        if not is_valid_file_content(response.content, response.headers.get("Content-Type", "")):
            return (
                f"AUTH/HTML response for {os.path.basename(urlparse(url).path)} — "
                "refresh cookies.txt (rtFa + FedAuth from same browser session)"
            )
        filename = os.path.basename(urlparse(url).path)
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "wb") as f:
            f.write(response.content)
        return f"Downloaded: {filepath} ({len(response.content)} bytes)"
    except Exception as e:
        return f"Error downloading {url}: {e}"


def copy_and_rename_one(row, source_dir, destination_dir):
    """Copy and rename a single file (used in parallel)."""
    old_filename = str(row['filename']).strip()
    new_filename = str(row['new name']).strip()
    file_ext = os.path.splitext(old_filename)[1].lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        return f"SKIP: {old_filename} (unsupported extension)"

    source_path = os.path.join(source_dir, old_filename)
    destination_path = os.path.join(destination_dir, new_filename)

    if not os.path.exists(source_path):
        # Case‑insensitive fallback
        found = None
        try:
            for actual in os.listdir(source_dir):
                if actual.lower() == old_filename.lower():
                    found = actual
                    source_path = os.path.join(source_dir, found)
                    break
        except Exception:
            pass
        if not found:
            return f"ERROR: {old_filename} not found"

    try:
        shutil.copy2(source_path, destination_path)
        if os.path.exists(destination_path):
            return f"SUCCESS: {old_filename} → {new_filename}"
        else:
            return f"ERROR: copy failed for {new_filename}"
    except Exception as e:
        return f"ERROR: {old_filename} - {e}"


def convert_pdf_to_jpeg_batched(pdf_path, dpi=150, pages_per_batch=10):
    """Convert a single PDF to JPEG (called per PDF in parallel)."""
    folder_path = os.path.dirname(pdf_path)
    filename = os.path.basename(pdf_path)
    base_name = os.path.splitext(filename)[0]
    try:
        info = pdfinfo_from_path(pdf_path, poppler_path=POPPLER_PATH)
        total_pages = info["Pages"]
    except Exception:
        total_pages = None

    page_start = 1
    all_converted = True
    while True:
        try:
            images = convert_from_path(
                pdf_path,
                dpi=dpi,
                poppler_path=POPPLER_PATH,
                first_page=page_start,
                last_page=min(page_start + pages_per_batch - 1, total_pages) if total_pages else None,
                thread_count=1
            )
            if not images:
                break
            for i, image in enumerate(images):
                page_num = page_start + i
                jpeg_file = f"{base_name}@page{page_num}.jpeg"
                jpeg_path = os.path.join(folder_path, jpeg_file)
                image.save(jpeg_path, 'JPEG', quality=85, optimize=True)
                del image
            page_start += len(images)
            del images
            gc.collect()
            if total_pages and page_start > total_pages:
                break
        except Exception as e:
            print(f"Error processing {filename} batch {page_start}: {e}")
            all_converted = False
            break
    if all_converted:
        os.remove(pdf_path)
        return f"Converted and removed: {filename}"
    else:
        return f"Partial conversion: {filename}"


def delete_duplicate_images(folder_path, dedup_folder):
    """Sequential deduplication using perceptual hashing."""
    image_hashes = {}
    os.makedirs(dedup_folder, exist_ok=True)
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.lower().endswith(('.jpeg', '.jpg', '.png')):
            try:
                with Image.open(file_path) as img:
                    img_hash = imagehash.phash(img)
                if img_hash in image_hashes:
                    print(f"Duplicate: {file_path} (original: {image_hashes[img_hash]})")
                else:
                    image_hashes[img_hash] = file_path
                    dest = os.path.join(dedup_folder, file_name)
                    os.rename(file_path, dest)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")


# ================================
# Main Pipeline
# ================================
def main():
    parser = argparse.ArgumentParser(description='Parallel image processing for P&P Round')
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--iteration', type=int, choices=[1, 2, 3, 4], required=True)
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Use files already in imgs_all; run rename, PDF conversion, and dedup only',
    )
    args = parser.parse_args()

    run_mode = args.iteration
    round_num = args.round

    global DOWNLOAD_HEADERS
    if not args.skip_download:
        DOWNLOAD_HEADERS, cookie_source = build_download_headers()
        print(f"Download auth: using {cookie_source}")
    else:
        print("Skipping download (--skip-download); using existing imgs_all/")

    # Define directories
    all_images = 'imgs_all'
    images_deduplicated = 'imgs_deduplicated'
    all_images_renamed = 'imgs_all_renamed'
    city_state_file = "City_State_key.xlsx"

    for d in [all_images, all_images_renamed, images_deduplicated]:
        os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data based on run_mode
    # ------------------------------------------------------------------
    if run_mode >= 2:
        prev_form_data = f"FormData-R{round_num} pt{run_mode - 1}.csv"
        latest_form_data = f"FormData-R{round_num} pt{run_mode}.csv"
        latest_df = pd.read_csv(latest_form_data)
        prev_df = pd.read_csv(prev_form_data)
        new_ids = list(set(latest_df['Result ID']) - set(prev_df['Result ID']))
        df_latest = latest_df[latest_df['Result ID'].isin(new_ids)]
        df_latest = df_latest[~((df_latest['Q2: Select outlet details for second visit'].isnull()) &
                                (df_latest['Q1: Enter Paints & Putty Visit Detail'] == 'Existing Outlet'))]
    else:
        form_data = f"FormData-R{round_num} pt{run_mode}.csv"
        df_latest = pd.read_csv(form_data)
        df_latest = df_latest[~((df_latest['Q2: Select outlet details for second visit'].isnull()) &
                                (df_latest['Q1: Enter Paints & Putty Visit Detail'] == 'Existing Outlet'))]

    # ------------------------------------------------------------------
    # Prepare URL list for download
    # ------------------------------------------------------------------
    df = df_latest.copy()
    df = df.rename(columns={
        'Q1: Enter Paints & Putty Visit Detail': 'Visit Detail',
        'Q2: Select outlet details for second visit': 'Outlet Details',
        'Q23: Name of Outlet Enter Details': 'Outlet Name',
        'Q22: Research Center Enter Details': 'City',
        'Q28: Outlet Code (Same as Result ID) Enter Details': 'Outlet Code',
        'Q14: If consent received for scanning invoices, please scan invoices for entire month': 'URL 1',
        'Q15: If consent received for uploading invoices, please upload invoices for entire month': 'URL 2'
    })
    df = df[['Result ID', 'Surveyed End Date', 'Visit Detail', 'Outlet Details',
             'URL 1', 'URL 2', 'City', 'Outlet Name', 'Outlet Code']]
    df['URL 1'] = df['URL 1'].apply(parse_url_cell)
    df['URL 2'] = df['URL 2'].apply(parse_url_cell)
    df['URL'] = df['URL 1'] + df['URL 2']
    df_filtered = df[df['URL'].apply(len) > 0][['Result ID', 'URL']]
    df_explode = df_filtered.explode('URL')
    df_links = df_explode

    if not args.skip_download:
        print(f"Downloading images using {NUM_WORKERS} workers...")
        tasks = [(row['URL'], all_images, idx) for idx, row in df_links.iterrows()]
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(download_image, url, out, i) for url, out, i in tasks]
            for future in as_completed(futures):
                print(future.result())
    else:
        n_existing = len([f for f in os.listdir(all_images) if os.path.isfile(os.path.join(all_images, f))])
        print(f"Skipped download; {n_existing} file(s) in {all_images}/")

    # ------------------------------------------------------------------
    # Prepare renaming mapping
    # ------------------------------------------------------------------
    df = df_latest.copy()
    df = df.rename(columns={
        'Q1: Enter Paints & Putty Visit Detail': 'Visit Detail',
        'Q2: Select outlet details for second visit': 'Outlet Details',
        'Q23: Name of Outlet Enter Details': 'Outlet Name',
        'Q22: Research Center Enter Details': 'City',
        'Q28: Outlet Code (Same as Result ID) Enter Details': 'Outlet Code',
        'Q14: If consent received for scanning invoices, please scan invoices for entire month': 'URL 1',
        'Q15: If consent received for uploading invoices, please upload invoices for entire month': 'URL 2'
    })
    df = df[~((df['Outlet Details'].isnull()) & (df['Visit Detail'] == 'Existing Outlet'))]
    df = df[['Result ID', 'Surveyed End Date', 'Visit Detail', 'Outlet Details',
             'URL 1', 'URL 2', 'City', 'Outlet Name', 'Outlet Code']]

    df['URL 1'] = df['URL 1'].apply(parse_url_cell)
    df['URL 2'] = df['URL 2'].apply(parse_url_cell)
    df['image name 1'] = df['URL 1'].apply(file_name)
    df['image name 2'] = df['URL 2'].apply(file_name)
    df['image name'] = df['image name 1'] + df['image name 2']

    df_filtered = df[df['image name'].apply(len) > 0]
    df_explode = df_filtered.explode('image name')
    df_explode.drop(columns=['URL 1', 'URL 2', 'image name 1', 'image name 2'], inplace=True)

    city_state_key = pd.read_excel(city_state_file)
    city_state_dict = dict(city_state_key.values)
    df_explode['State'] = df_explode['City'].apply(lambda x: city_state_dict[x] if pd.notna(x) else x)
    source = df_explode

    source['Surveyed End Date'] = pd.to_datetime(source['Surveyed End Date'], format='mixed', dayfirst=True, errors='coerce')
    source['Surveyed End Date'] = source['Surveyed End Date'].dt.strftime('%d%b%y').str.lower()

    source['State'] = source[['State', 'Outlet Details', 'Visit Detail']].apply(
        lambda x: x[1].split('_')[0] if x[2] == 'Existing Outlet' else x[0], axis=1)
    source['City'] = source[['City', 'Outlet Details', 'Visit Detail']].apply(
        lambda x: x[1].split('_')[1] if x[2] == 'Existing Outlet' else x[0], axis=1)
    source['Outlet Name'] = source[['Outlet Name', 'Outlet Details', 'Visit Detail']].apply(
        lambda x: x[1].split('_')[2] if x[2] == 'Existing Outlet' else x[0], axis=1)
    source['Outlet Code'] = source[['Outlet Code', 'Outlet Details', 'Visit Detail']].apply(
        lambda x: x[1].split('_')[3] if x[2] == 'Existing Outlet' else x[0], axis=1)

    source['Outlet Name updated'] = source['Outlet Name'].apply(lambda x: str(x).replace('/', '').replace('\n', ''))
    source['Outlet Details final'] = source[['State', 'City', 'Outlet Name updated', 'Outlet Code']].apply(
        lambda x: f"{x[0]}_{x[1]}_{x[2]}_{x[3]}", axis=1)
    source = source.drop_duplicates(subset='image name')

    df_images = pd.DataFrame(os.listdir(all_images), columns=['filename'])
    df_match = df_images.merge(source, left_on='filename', right_on='image name', how='left')
    df_match = df_match[df_match['Outlet Details final'].notnull()]

    df_match['file ext'] = df_match['image name'].apply(lambda x: x.split('.')[-1])
    df_match['image name'] = df_match['image name'].apply(lambda x: x.split('.')[0])
    df_match['updated image name'] = df_match[['image name', 'file ext']].apply(
        lambda x: '_'.join(x[0].split('_')[0:2]) + str(random.randint(0, 100000)) + '.' + x[1], axis=1)
    df_match['new name'] = df_match[['Surveyed End Date', 'Outlet Details final', 'Result ID', 'updated image name']].apply(
        lambda x: f"{x[0]}_{x[1]}_nan_{x[2]}_{x[3]}", axis=1)
    file_name_change = df_match[['filename', 'new name']]
    file_name_change.to_csv('updated_filename_dict.csv', index=False)

    # --- Parallel copy & rename ---
    print(f"Copying and renaming files using {NUM_WORKERS} workers...")
    rows = [row for _, row in file_name_change.iterrows()]
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(copy_and_rename_one, row, all_images, all_images_renamed) for row in rows]
        for future in as_completed(futures):
            print(future.result())

    # --- Parallel PDF conversion (one thread per PDF) ---
    print(f"Converting PDFs to JPEG using {NUM_WORKERS} workers...")
    pdf_files = [os.path.join(all_images_renamed, f) for f in os.listdir(all_images_renamed) if f.lower().endswith('.pdf')]
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(convert_pdf_to_jpeg_batched, pdf_path) for pdf_path in pdf_files]
        for future in as_completed(futures):
            print(future.result())

    # --- Sequential deduplication ---
    print("Removing duplicate images (sequential)...")
    delete_duplicate_images(all_images_renamed, images_deduplicated)

    print("\nAll processing completed.")


if __name__ == "__main__":
    main()