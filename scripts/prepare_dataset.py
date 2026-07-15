#!/usr/bin/env python3
"""
RxVision Dataset Preparation Pipeline

Parses an NLM C3PI-format reference table, matches locally obtained images, normalizes drug names,
performs stratified train/val/test splitting, and outputs a PyTorch ImageFolder-ready
directory structure.

This is a research data-engineering utility, not a benchmark builder. Its
image-level split can leak near-duplicate views across partitions and requires
an explicit acknowledgement before it copies images.

Usage:
    python scripts/prepare_dataset.py --help
    python scripts/prepare_dataset.py --min-images 30 --max-classes 50
    python scripts/prepare_dataset.py --list-drugs --min-images 50
    python scripts/prepare_dataset.py --drugs "ALLOPURINOL" "GLIMEPIRIDE"
    python scripts/prepare_dataset.py --ndcs 00093725401 00591554405
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Fail closed on truncated images.
ImageFile.LOAD_TRUNCATED_IMAGES = False

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_file: str | None = None, verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode="a"))
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    return logging.getLogger("prepare_dataset")


# ---------------------------------------------------------------------------
# Drug name normalization
# ---------------------------------------------------------------------------

# Dosage-form suffixes to strip (order matters: longer first)
_FORM_SUFFIXES = [
    "TABLETS, USP",
    "TABLETS USP",
    "CAPSULES, USP",
    "CAPSULES USP",
    "TABLETS",
    "TABLET",
    "CAPSULES",
    "CAPSULE",
    "TABS",
    "TAB",
    "CAPS",
    "CAP",
    "ORAL SOLUTION",
    "ORAL SUSPENSION",
    "INJECTION",
    "SOLUTION",
    "SUSPENSION",
    "CREAM",
    "OINTMENT",
    "GEL",
    "LOTION",
    "PATCH",
    "SPRAY",
    "DROPS",
    "SYRUP",
    "ELIXIR",
    "POWDER",
    "GRANULES",
    "SUPPOSITORY",
    "INHALER",
    "CHEWABLE",
    "EXTENDED-RELEASE",
    "EXTENDED RELEASE",
    "ER",
    "XR",
    "SR",
    "DR",
    "CR",
    "LA",
    "USP",
]

# Pre-compile a regex that matches any suffix at end of string (after optional whitespace)
_SUFFIX_PATTERN = re.compile(
    r"\s+(?:" + "|".join(re.escape(s) for s in _FORM_SUFFIXES) + r")\s*$",
    re.IGNORECASE,
)


def normalize_drug_name(name: str) -> str:
    """Normalize a drug name for consistent grouping.

    Steps:
      1. Uppercase and strip whitespace
      2. Remove trailing dosage-form suffixes (TAB, TABLETS USP, etc.)
      3. Collapse multiple spaces
      4. Strip trailing commas/spaces
    """
    if not isinstance(name, str):
        return "UNKNOWN"
    name = name.strip().upper()

    # Iteratively strip suffixes (some names have stacked suffixes)
    prev = None
    while name != prev:
        prev = name
        name = _SUFFIX_PATTERN.sub("", name)

    # Remove trailing commas, periods, whitespace
    name = re.sub(r"[,.\s]+$", "", name)
    # Collapse whitespace
    name = re.sub(r"\s+", " ", name).strip()
    return name if name else "UNKNOWN"


def make_folder_safe(name: str) -> str:
    """Convert a drug name to a filesystem-safe folder name."""
    safe = name.lower().strip()
    safe = re.sub(r"[^a-z0-9]+", "_", safe)
    safe = safe.strip("_")
    return safe if safe else "unknown"


# ---------------------------------------------------------------------------
# Reference table parsing
# ---------------------------------------------------------------------------


def load_reference_table(path: str, logger: logging.Logger) -> pd.DataFrame:
    """Parse the NIH C3PI directory_of_images.txt reference table.

    Format: NDC|PART|FILE_PATH|TYPE|DRUG_NAME
    """
    logger.info(f"Loading reference table from {path}")

    df = pd.read_csv(
        path,
        sep="|",
        header=None,
        names=["ndc", "part", "file_path", "type", "drug_name"],
        dtype={"ndc": str},
    )

    initial_count = len(df)
    df = df.dropna(subset=["ndc", "file_path", "drug_name"])
    dropped = initial_count - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped} rows with missing values")

    # Uppercase drug names
    df["drug_name"] = df["drug_name"].str.upper().str.strip()

    # Extract filename from file_path (e.g. PillProjectDisc69/images/FILENAME.EXT)
    df["filename"] = df["file_path"].apply(
        lambda p: p.rsplit("/", 1)[-1] if isinstance(p, str) else ""
    )

    # Extract file extension
    df["ext"] = df["filename"].apply(lambda f: os.path.splitext(f)[1].upper())

    # Filter out video files
    video_exts = {".WMV", ".AVI", ".MP4", ".MOV"}
    video_mask = df["ext"].isin(video_exts)
    n_videos = video_mask.sum()
    if n_videos:
        logger.info(f"Filtering out {n_videos} video files")
        df = df[~video_mask]

    # Normalize drug names
    df["drug_normalized"] = df["drug_name"].apply(normalize_drug_name)
    df["folder_name"] = df["drug_normalized"].apply(make_folder_safe)

    logger.info(
        f"Loaded {len(df):,} image records | "
        f"{df['ndc'].nunique():,} unique NDCs | "
        f"{df['drug_normalized'].nunique():,} unique drugs (normalized)"
    )

    return df


# ---------------------------------------------------------------------------
# Image discovery and matching
# ---------------------------------------------------------------------------


def discover_raw_images(raw_dir: str, logger: logging.Logger) -> dict[str, Path]:
    """Walk the raw images directory and return {filename: full_path}.

    Handles nested structures like PillProjectDisc*/images/*.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        logger.warning(f"Raw images directory does not exist: {raw_dir}")
        return {}

    logger.info(f"Scanning raw images in {raw_dir} ...")
    image_exts = {".jpg", ".jpeg", ".png", ".cr2", ".tiff", ".tif", ".bmp"}
    index: dict[str, Path] = {}
    file_count = 0

    for root, _dirs, files in os.walk(raw_path):
        for fname in files:
            file_count += 1
            if os.path.splitext(fname)[1].lower() in image_exts:
                # Use filename as key (matches reference table)
                index[fname] = Path(root) / fname

    logger.info(f"Scanned {file_count:,} files, indexed {len(index):,} images")
    return index


def match_images(
    df: pd.DataFrame, image_index: dict[str, Path], logger: logging.Logger
) -> pd.DataFrame:
    """Cross-reference the reference table against locally available images."""
    df = df.copy()
    df["local_path"] = df["filename"].map(image_index)
    matched = df["local_path"].notna()

    # Also try matching without extension (for CR2 -> JPG conversions already done)
    if matched.sum() < len(df):
        stem_index: dict[str, Path] = {}
        for fname, fpath in image_index.items():
            stem = os.path.splitext(fname)[0]
            if stem not in stem_index:
                stem_index[stem] = fpath

        unmatched = df[~matched]
        stems = unmatched["filename"].apply(lambda f: os.path.splitext(f)[0])
        alt_paths = stems.map(stem_index)
        df.loc[~matched, "local_path"] = alt_paths

    total_matched = df["local_path"].notna().sum()
    logger.info(
        f"Matched {total_matched:,} / {len(df):,} reference rows to local files "
        f"({total_matched / len(df) * 100:.1f}%)"
    )

    return df[df["local_path"].notna()].copy()


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

# Try to import rawpy for CR2 conversion
try:
    import rawpy

    HAS_RAWPY = True
except ImportError:
    HAS_RAWPY = False


def validate_image(path: Path) -> bool:
    """Return True if the image can be opened by PIL."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def process_image(
    src: Path,
    dst: Path,
    size: int = 224,
    logger: logging.Logger | None = None,
) -> bool:
    """Process a single image: convert CR2 if needed, resize, save as JPG.

    Returns True on success.
    """
    try:
        ext = src.suffix.upper()

        if ext == ".CR2":
            if not HAS_RAWPY:
                if logger:
                    logger.debug(f"Skipping CR2 (rawpy not available): {src.name}")
                return False
            raw = rawpy.imread(str(src))
            rgb = raw.postprocess()
            img = Image.fromarray(rgb)
        else:
            img = Image.open(src)

        # Convert to RGB (handles RGBA, palette, grayscale)
        img = img.convert("RGB")

        # Resize with high-quality resampling
        img = img.resize((size, size), Image.LANCZOS)

        # Ensure parent directory exists
        dst.parent.mkdir(parents=True, exist_ok=True)

        img.save(dst, "JPEG", quality=95)
        return True

    except Exception as e:
        if logger:
            logger.debug(f"Failed to process {src.name}: {e}")
        return False


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform stratified train/val/test split ensuring every class is represented.

    For classes with very few samples (< 3), images are distributed round-robin.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    # Separate small classes that can't be stratified normally
    class_counts = df["folder_name"].value_counts()
    small_classes = set(class_counts[class_counts < 3].index)

    large_df = df[~df["folder_name"].isin(small_classes)]
    small_df = df[df["folder_name"].isin(small_classes)]

    train_parts, val_parts, test_parts = [], [], []

    # Stratified split for classes with enough samples
    if len(large_df) > 0:
        # First split: train vs (val+test)
        val_test_ratio = val_ratio + test_ratio
        train_df, valtest_df = train_test_split(
            large_df,
            test_size=val_test_ratio,
            stratify=large_df["folder_name"],
            random_state=random_state,
        )

        # Second split: val vs test
        relative_test = test_ratio / val_test_ratio
        val_df, test_df = train_test_split(
            valtest_df,
            test_size=relative_test,
            stratify=valtest_df["folder_name"],
            random_state=random_state,
        )

        train_parts.append(train_df)
        val_parts.append(val_df)
        test_parts.append(test_df)

    # Round-robin for small classes
    if len(small_df) > 0:
        logger.info(f"Round-robin splitting {len(small_classes)} classes with <3 images")
        for _cls, group in small_df.groupby("folder_name"):
            rows = group.sample(frac=1, random_state=random_state).reset_index(drop=True)
            n = len(rows)
            # Always put at least one in train
            if n == 1:
                train_parts.append(rows)
            elif n == 2:
                train_parts.append(rows.iloc[:1])
                val_parts.append(rows.iloc[1:2])
            else:
                # Should not happen since we filtered <3, but handle gracefully
                train_parts.append(rows.iloc[:1])
                val_parts.append(rows.iloc[1:2])
                test_parts.append(rows.iloc[2:])

    train = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
    val = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame()
    test = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()

    logger.info(f"Split sizes — train: {len(train):,} | val: {len(val):,} | test: {len(test):,}")

    return train, val, test


# ---------------------------------------------------------------------------
# Class selection
# ---------------------------------------------------------------------------


def select_classes(
    df: pd.DataFrame,
    min_images: int,
    max_classes: int | None,
    max_per_class: int | None,
    group_by: str,
    specific_ndcs: list[str] | None,
    specific_drugs: list[str] | None,
    image_types: list[str] | None,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Filter and select classes based on criteria."""
    result = df.copy()

    # Filter by image type if specified
    if image_types:
        result = result[result["type"].isin(image_types)]
        logger.info(f"Filtered to {len(result):,} rows matching types: {image_types}")

    # Filter by specific NDCs
    if specific_ndcs:
        result = result[result["ndc"].isin(specific_ndcs)]
        logger.info(f"Filtered to {len(result):,} rows for {len(specific_ndcs)} NDCs")
        return _apply_per_class_cap(result, max_per_class, group_by, logger)

    # Filter by specific drug names (match against normalized names)
    if specific_drugs:
        drug_set = {normalize_drug_name(d) for d in specific_drugs}
        result = result[result["drug_normalized"].isin(drug_set)]
        logger.info(f"Filtered to {len(result):,} rows for drugs: {drug_set}")
        return _apply_per_class_cap(result, max_per_class, group_by, logger)

    # Group by the chosen column
    group_col = "folder_name" if group_by == "drug" else "ndc"
    counts = result[group_col].value_counts()

    # Apply minimum image threshold
    passing = counts[counts >= min_images]
    logger.info(f"{len(passing):,} classes with >= {min_images} images (of {len(counts):,} total)")

    if max_classes and len(passing) > max_classes:
        # Take the top N classes by image count
        passing = passing.head(max_classes)
        logger.info(f"Capped to top {max_classes} classes by image count")

    result = result[result[group_col].isin(passing.index)]

    return _apply_per_class_cap(result, max_per_class, group_col, logger)


def _apply_per_class_cap(
    df: pd.DataFrame,
    max_per_class: int | None,
    group_col: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Optionally cap the number of images per class for balance."""
    if max_per_class is None:
        return df

    before = len(df)
    df = df.groupby(group_col, group_keys=False).apply(
        lambda g: g.sample(n=min(len(g), max_per_class), random_state=42)
    )
    after = len(df)
    if before != after:
        logger.info(f"Capped per-class images to {max_per_class}: {before:,} -> {after:,}")
    return df


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------


def load_progress(progress_file: Path) -> set[str]:
    """Load the set of already-processed filenames."""
    if progress_file.exists():
        with open(progress_file) as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def save_progress(progress_file: Path, processed: set[str]) -> None:
    """Save processed filenames for resumability."""
    with open(progress_file, "w") as f:
        f.write("\n".join(sorted(processed)))


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def compute_and_save_stats(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> dict:
    """Compute dataset statistics and save to JSON."""
    stats: dict = {"splits": {}}

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if split_df.empty:
            stats["splits"][split_name] = {"total": 0, "classes": 0}
            continue
        class_counts = split_df["folder_name"].value_counts()
        stats["splits"][split_name] = {
            "total": len(split_df),
            "classes": len(class_counts),
            "min_per_class": int(class_counts.min()),
            "max_per_class": int(class_counts.max()),
            "mean_per_class": round(float(class_counts.mean()), 1),
            "median_per_class": int(class_counts.median()),
        }

    # Overall
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    overall_counts = all_df["folder_name"].value_counts()
    stats["overall"] = {
        "total_images": len(all_df),
        "total_classes": len(overall_counts),
        "images_per_class": {k: int(v) for k, v in overall_counts.items()},
    }

    stats_path = output_dir / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Dataset statistics saved to {stats_path}")

    # Print summary
    logger.info("=" * 60)
    logger.info("DATASET STATISTICS")
    logger.info("=" * 60)
    logger.info(f"  Total images:   {stats['overall']['total_images']:,}")
    logger.info(f"  Total classes:  {stats['overall']['total_classes']}")
    for split_name in ["train", "val", "test"]:
        s = stats["splits"][split_name]
        logger.info(
            f"  {split_name:>5}: {s['total']:>6,} images across {s.get('classes', 0)} classes"
        )
    if overall_counts.size > 0:
        logger.info(f"  Min images/class:    {overall_counts.min()}")
        logger.info(f"  Max images/class:    {overall_counts.max()}")
        logger.info(f"  Median images/class: {int(overall_counts.median())}")
    logger.info("=" * 60)

    return stats


def generate_class_mapping(df: pd.DataFrame, output_dir: Path, logger: logging.Logger) -> None:
    """Generate class_names.json for model consumption."""
    # Get unique classes sorted alphabetically
    classes = sorted(df["folder_name"].unique())

    # Build NDC -> normalized name mapping
    ndc_to_name: dict[str, str] = {}
    for _, row in df[["ndc", "drug_normalized"]].drop_duplicates().iterrows():
        ndc_to_name[row["ndc"]] = row["drug_normalized"]

    # Build name -> index mapping
    name_to_idx = {name: idx for idx, name in enumerate(classes)}

    # Build folder -> display name mapping
    folder_to_display: dict[str, str] = {}
    for _, row in df[["folder_name", "drug_normalized"]].drop_duplicates().iterrows():
        folder_to_display[row["folder_name"]] = row["drug_normalized"]

    mapping = {
        "classes": classes,
        "num_classes": len(classes),
        "ndc_to_name": ndc_to_name,
        "name_to_idx": name_to_idx,
        "folder_to_display_name": folder_to_display,
    }

    mapping_path = output_dir / "class_names.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)

    logger.info(f"Class mapping saved to {mapping_path} ({len(classes)} classes)")


# ---------------------------------------------------------------------------
# List drugs command
# ---------------------------------------------------------------------------


def list_drugs(df: pd.DataFrame, min_images: int, logger: logging.Logger) -> None:
    """Print available drugs and their image counts."""
    counts = df.groupby("drug_normalized").agg(
        images=("filename", "count"),
        ndcs=("ndc", "nunique"),
        original_names=("drug_name", lambda x: sorted(set(x))),
    )
    counts = counts.sort_values("images", ascending=False)

    if min_images > 0:
        counts = counts[counts["images"] >= min_images]

    print(f"\n{'Drug Name (Normalized)':<45} {'Images':>8} {'NDCs':>6}  Original Names")
    print("-" * 120)
    for name, row in counts.iterrows():
        originals = ", ".join(row["original_names"][:3])
        if len(row["original_names"]) > 3:
            originals += f" (+{len(row['original_names']) - 3} more)"
        print(f"{name:<45} {row['images']:>8,} {row['ndcs']:>6}  {originals}")

    print(f"\nTotal: {len(counts):,} drugs with >= {min_images} images")
    print(f"Total images: {counts['images'].sum():,}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full dataset preparation pipeline."""
    logger = setup_logging(log_file=args.log_file, verbose=args.verbose)
    logger.info("RxVision Dataset Preparation Pipeline")
    logger.info(f"Arguments: {vars(args)}")

    # Resolve paths relative to project root
    project_root = Path(args.project_root)
    ref_table_path = project_root / args.reference_table
    raw_dir = project_root / args.raw_dir
    output_dir = project_root / args.output_dir

    # 1. Load reference table
    df = load_reference_table(str(ref_table_path), logger)

    # Handle --list-drugs
    if args.list_drugs:
        list_drugs(df, args.min_images, logger)
        return

    # 2. Discover local raw images
    image_index = discover_raw_images(str(raw_dir), logger)

    if not image_index:
        logger.warning(
            "No raw images found. Obtain data independently, review its terms, "
            "and pass the local directory with --raw-dir."
        )
        # Continue anyway — useful for dry-run / planning

    # 3. Match reference table rows to local images
    if image_index:
        if not args.allow_image_level_split:
            raise ValueError(
                "Image-level splitting can leak near-duplicate views. Pass "
                "--allow-image-level-split only for pipeline development; do not "
                "use the resulting metrics as a defensible benchmark."
            )
        df = match_images(df, image_index, logger)
        if df.empty:
            logger.error("No images matched between reference table and raw directory.")
            sys.exit(1)
    else:
        logger.info("No raw images available — running in dry-run mode (stats only)")

    # 4. Select classes
    df = select_classes(
        df,
        min_images=args.min_images,
        max_classes=args.max_classes,
        max_per_class=args.max_per_class,
        group_by=args.group_by,
        specific_ndcs=args.ndcs,
        specific_drugs=args.drugs,
        image_types=args.types,
        logger=logger,
    )

    if df.empty:
        logger.error("No classes remaining after filtering. Adjust your criteria.")
        sys.exit(1)

    logger.info(f"Selected {df['folder_name'].nunique()} classes with {len(df):,} total images")

    # 5. Stratified split
    train_ratio, val_ratio, test_ratio = args.split
    train_df, val_df, test_df = stratified_split(
        df, train_ratio, val_ratio, test_ratio, args.seed, logger
    )

    # 6. Process and copy images into output structure
    if image_index:
        _process_splits(
            splits={"train": train_df, "val": val_df, "test": test_df},
            output_dir=output_dir,
            image_size=args.image_size,
            logger=logger,
        )
    else:
        logger.info("Skipping image processing (dry-run mode, no raw images)")

    # 7. Generate class mapping
    generate_class_mapping(
        pd.concat([train_df, val_df, test_df], ignore_index=True),
        output_dir,
        logger,
    )

    # 8. Compute and save statistics
    compute_and_save_stats(train_df, val_df, test_df, output_dir, logger)

    logger.info("Pipeline complete.")


def _process_splits(
    splits: dict[str, pd.DataFrame],
    output_dir: Path,
    image_size: int,
    logger: logging.Logger,
) -> None:
    """Process images into train/val/test directory structure."""
    progress_file = output_dir / ".prepare_progress"
    processed = load_progress(progress_file)
    logger.info(f"Previously processed: {len(processed):,} images")

    total = sum(len(s) for s in splits.values())
    succeeded = 0
    skipped = 0
    failed = 0

    with tqdm(total=total, desc="Processing images", unit="img") as pbar:
        for split_name, split_df in splits.items():
            split_dir = output_dir / split_name

            for _, row in split_df.iterrows():
                src = Path(row["local_path"])
                # Destination: data/{split}/{folder_name}/{stem}.jpg
                dst_stem = os.path.splitext(src.name)[0]
                dst = split_dir / row["folder_name"] / f"{dst_stem}.jpg"

                # Unique key for progress tracking
                progress_key = f"{split_name}/{row['folder_name']}/{dst_stem}.jpg"

                if progress_key in processed:
                    skipped += 1
                    pbar.update(1)
                    continue

                if dst.exists():
                    # Already exists on disk but not in progress file
                    processed.add(progress_key)
                    skipped += 1
                    pbar.update(1)
                    continue

                ok = process_image(src, dst, size=image_size, logger=logger)
                if ok:
                    succeeded += 1
                    processed.add(progress_key)
                else:
                    failed += 1

                pbar.update(1)

                # Periodically save progress (every 500 images)
                if (succeeded + failed) % 500 == 0:
                    save_progress(progress_file, processed)

    save_progress(progress_file, processed)

    logger.info(
        f"Image processing complete — "
        f"succeeded: {succeeded:,} | skipped: {skipped:,} | failed: {failed:,}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RxVision Dataset Preparation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s --list-drugs                          # Show all available drugs
  %(prog)s --list-drugs --min-images 50           # Drugs with 50+ images
  %(prog)s --min-images 30 --max-classes 50       # Top 50 classes with 30+ images
  %(prog)s --ndcs 00093725401 00591554405         # Specific NDCs
  %(prog)s --drugs "ALLOPURINOL" "GLIMEPIRIDE"    # Specific drugs
  %(prog)s --types C3PI_Reference C3PI_Test       # Filter by image type
  %(prog)s --max-per-class 50                     # Cap for class balance
""",
    )

    # Class selection
    sel = parser.add_argument_group("Class selection")
    sel.add_argument(
        "--min-images",
        type=int,
        default=30,
        help="Minimum images per class to include (default: 30)",
    )
    sel.add_argument(
        "--max-classes",
        type=int,
        default=None,
        help="Maximum number of classes to include (top N by count)",
    )
    sel.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Cap images per class for balance (e.g. median count)",
    )
    sel.add_argument(
        "--ndcs",
        nargs="+",
        default=None,
        help="Specific NDC codes to include",
    )
    sel.add_argument(
        "--drugs",
        nargs="+",
        default=None,
        help="Specific drug names to include (matched against normalized names)",
    )
    sel.add_argument(
        "--types",
        nargs="+",
        default=None,
        help="Image types to include (e.g. C3PI_Reference, C3PI_Test)",
    )
    sel.add_argument(
        "--group-by",
        choices=["drug", "ndc"],
        default="drug",
        help="Group images by drug name or NDC code (default: drug)",
    )

    # Splitting
    spl = parser.add_argument_group("Dataset splitting")
    spl.add_argument(
        "--split",
        nargs=3,
        type=float,
        default=[0.7, 0.15, 0.15],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train/val/test split ratios (default: 0.7 0.15 0.15)",
    )
    spl.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # Image processing
    img = parser.add_argument_group("Image processing")
    img.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Output image size in pixels (default: 224)",
    )

    # Paths
    paths = parser.add_argument_group("Paths")
    paths.add_argument(
        "--project-root",
        type=str,
        default=os.environ.get("RXVISION_ROOT", "."),
        help="Project root directory (default: cwd or $RXVISION_ROOT)",
    )
    paths.add_argument(
        "--reference-table",
        type=str,
        required=True,
        help="Path to an independently obtained reference table",
    )
    paths.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Path to raw downloaded images (relative to project root)",
    )
    paths.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for train/val/test (relative to project root)",
    )

    # Listing
    lst = parser.add_argument_group("Listing")
    lst.add_argument(
        "--list-drugs",
        action="store_true",
        help="List available drugs and image counts, then exit",
    )

    # Misc
    misc = parser.add_argument_group("Miscellaneous")
    misc.add_argument(
        "--allow-image-level-split",
        action="store_true",
        help=(
            "Acknowledge that image-level splits may leak near-duplicates and are "
            "for pipeline development, not performance claims"
        ),
    )
    misc.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    misc.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (optional)",
    )

    args = parser.parse_args(argv)

    # Validate split ratios
    if abs(sum(args.split) - 1.0) > 1e-6:
        parser.error(f"Split ratios must sum to 1.0, got {sum(args.split):.4f}")

    return args


def main() -> None:
    args = parse_args()
    try:
        run_pipeline(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        logging.getLogger("prepare_dataset").error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
