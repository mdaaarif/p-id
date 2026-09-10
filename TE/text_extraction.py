import fitz
import pandas as pd
import os
import glob
import re
import math


class TextExtractor:

    def __init__(self, config):

        """
        Initialize the TextExtractor with a configuration object.

        :param config: Configuration module containing all path and setting constants
        :type  config: module

        :return: None
        :rtype : None

        :raises AttributeError: When config is missing required directory attributes
        """

        self.config = config

        self.all_process_tags = []
        self.all_equipment_info = []

        self._create_directories()

    def _create_directories(self):

        """
        Create all required output directories if they do not already exist.

        :param self: Instance of TextExtractor carrying the config object
        :type  self: TextExtractor

        :return: None

        :raises OSError: When directory creation fails due to permission or path issues
        """

        for d in [
            self.config.RESULTS_CSV_DIR,
            self.config.RESULTS_JSON_DIR,
            self.config.RESULTS_IMAGES_DIR,
            self.config.PER_PDF_DIR
        ]:
            os.makedirs(d, exist_ok=True)

    def extract_equipment_id(self, filename):

        """
        Extract the equipment ID from a PDF filename by stripping its extension and path.

        :param self    : Instance of TextExtractor
        :type  self    : TextExtractor
        :param filename: Full or relative path to the PDF file
        :type  filename: str

        :return: Equipment ID string derived from the filename without extension
        :rtype : str
        """

        return os.path.splitext(
            os.path.basename(filename)
        )[0]

    def extract_text_from_pdf(self, pdf_path):

        """
        Open a PDF and extract all text spans with their bounding boxes into a DataFrame.
        Bounding box coordinates are rounded to 4 decimal places.

        :param self    : Instance of TextExtractor
        :type  self    : TextExtractor
        :param pdf_path: Absolute or relative path to the input PDF file
        :type  pdf_path: str

        :return: Tuple of (DataFrame with columns [equipment_number, page, text, bbox],
                 equipment ID string)
        :rtype : tuple[pd.DataFrame, str]

        :raises FileNotFoundError : When the PDF file does not exist at the given path
        :raises fitz.FileDataError: When the file exists but cannot be opened as a valid PDF
        """

        doc = fitz.open(pdf_path)

        rows = []

        eq = self.extract_equipment_id(pdf_path)

        for page_no, page in enumerate(doc):

            data = page.get_text("dict")

            for block in data["blocks"]:

                if "lines" not in block:
                    continue

                for line in block["lines"]:

                    for span in line["spans"]:
                        rows.append(
                            {
                                "equipment_number": eq,
                                "page": page_no + 1,
                                "text": span["text"],
                                "bbox": (
                                    round(float(span["bbox"][0]), 4),
                                    round(float(span["bbox"][1]), 4),
                                    round(float(span["bbox"][2]), 4),
                                    round(float(span["bbox"][3]), 4)
                                )
                            }
                        )

        doc.close()

        return pd.DataFrame(rows), eq

    def is_pure_symbol(self, text):

        """
        Determine whether a text string contains no meaningful alphanumeric content
        and should be treated as a pure symbol or whitespace.

        :param self: Instance of TextExtractor
        :type  self: TextExtractor
        :param text: Raw text string extracted from a PDF span
        :type  text: str

        :return: True if the text is empty, a single non-alphanumeric character,
                 contains no alphanumeric characters, or has an alphanumeric ratio
                 below 0.30 for strings longer than 2 characters; False otherwise
        :rtype : bool
        """

        t = str(text).strip()

        if t == "":
            return True

        if len(t) == 1 and not t.isalnum():
            return True

        alpha_num = sum(1 for c in t if c.isalnum())

        if alpha_num == 0:
            return True

        if len(t) > 2:
            ratio = alpha_num / len(t)
            if ratio < 0.3:
                return True

        return False

    def is_unwanted(self, text):

        """
        Determine whether a text string should be excluded from extraction output.
        Filters grid labels, scale annotations, measurement strings, margin metadata,
        notes references, and pure symbols.

        :param self: Instance of TextExtractor
        :type  self: TextExtractor
        :param text: Raw or merged text string to evaluate
        :type  text: str

        :return: True if the text matches any unwanted pattern (empty, pure symbol,
                 contains #, NOTE- prefix, scale ratio, measurement, single uppercase
                 letter, 1–2 digit number, or margin document reference); False otherwise
        :rtype : bool
        """

        t = str(text).strip()

        if t == "":
            return True

        if self.is_pure_symbol(t):
            return True

        if "#" in t:
            return True

        if re.match(r"^NOTE-.*", t, re.IGNORECASE):
            return True

        scale_patterns = [
            r"^SCALE\s*[:=]?\s*\d+\s*[:/]\s*\d+$",
            r"^\d+\s*[:/]\s*\d+$"
        ]

        for p in scale_patterns:
            if re.fullmatch(p, t, re.IGNORECASE):
                return True

        measurement_patterns = [
            r"^\d+\s*/\s*\d+\s*['\"]+$",
            r"^\d+(\.\d+)?\s*['\"]+$",
            r"^\d+(\.\d+)?\s*['\"]+\s*[xX×]\s*\d+(\.\d+)?\s*['\"]?$",
            r"^\d+(\.\d+)?\s*(mm|cm)$",
            r"^\d+\s*(mm|cm)?\s*[xX×]\s*\d+\s*(mm|cm)?$",
            r"^\d+\"\s*[xX×]\s*\d+(?:-\d+/\d+)?\"\s*$"
        ]

        for p in measurement_patterns:
            if re.fullmatch(p, t, re.IGNORECASE):
                return True

        if re.fullmatch(r"[A-Z]", t):
            return True

        if re.fullmatch(r"\d{1,2}", t):
            return True

        margin_patterns = [
            r"^[A-Z]\d{3}-[A-Z]{3}-[A-Z]{3}-\d{3}-\d{4}.*",
            r"^Rev\.\s*\d+",
            r"^Page\s+\d+\s+of\s+\d+",
            r"^\d{2}-[A-Z]{3}-\d{4}",
            r"^Doc\.",
            r"^Size\s+[A-Z0-9]+"
        ]

        for p in margin_patterns:
            if re.match(p, t, re.IGNORECASE):
                return True

        return False


    def parse_bbox(self, b):

        """
        Convert a bounding box sequence into a tuple of four Python floats.

        :param self: Instance of TextExtractor
        :type  self: TextExtractor
        :param b   : Sequence of four numeric values (x0, y0, x1, y1)
        :type  b   : tuple | list

        :return: Four-element tuple of floats (x0, y0, x1, y1)
        :rtype : tuple[float, float, float, float]

        :raises IndexError: When b contains fewer than four elements
        :raises ValueError: When any element of b cannot be converted to float
        """

        return (
            float(b[0]),
            float(b[1]),
            float(b[2]),
            float(b[3])
        )

    def add_bbox_columns(self, df):

        """
        Expand the bbox tuple column into individual coordinate and centre columns
        and append them to the DataFrame in-place.

        :param self: Instance of TextExtractor
        :type  self: TextExtractor
        :param df  : DataFrame containing a 'bbox' column of (x0, y0, x1, y1) tuples
        :type  df  : pd.DataFrame

        :return: DataFrame with six additional columns: x0, y0, x1, y1, center_x, center_y
        :rtype : pd.DataFrame

        :raises KeyError  : When the DataFrame is missing the 'bbox' column
        :raises IndexError: When any bbox tuple contains fewer than four elements
        """

        df["x0"] = df.bbox.apply(lambda x: x[0])
        df["y0"] = df.bbox.apply(lambda x: x[1])
        df["x1"] = df.bbox.apply(lambda x: x[2])
        df["y1"] = df.bbox.apply(lambda x: x[3])

        df["center_x"] = (df.x0 + df.x1) / 2
        df["center_y"] = (df.y0 + df.y1) / 2

        return df

    def remove_notes(self, df):

        """
        Remove rows whose bounding box centre falls within the notes region defined
        by the top-left corner of the drawing.

        :param self: Instance of TextExtractor
        :type  self: TextExtractor
        :param df  : DataFrame with center_x and center_y coordinate columns
        :type  df  : pd.DataFrame

        :return: Copy of the DataFrame with notes-region rows removed
        :rtype : pd.DataFrame

        :raises KeyError: When the DataFrame is missing 'center_x' or 'center_y' columns
        """

        df = df.copy()

        notes_filter = (
            (df["center_x"] < self.config.NOTES_MAX_X)
            &
            (df["center_y"] < self.config.NOTES_MAX_Y)
        )

        removed = df[notes_filter]

        print(f"Notes removed: {len(removed)}")

        return df[~notes_filter].copy()

    def split_equipment(self, df):

        """
        Partition a DataFrame into equipment info and process tag subsets based on
        the bounding box centre position relative to the title block boundary.

        :param self: Instance of TextExtractor
        :type  self: TextExtractor
        :param df  : DataFrame with center_x and center_y coordinate columns
        :type  df  : pd.DataFrame

        :return: Tuple of (equipment_df, process_df) where equipment_df contains rows
                 within the title block region and process_df contains all remaining rows
        :rtype : tuple[pd.DataFrame, pd.DataFrame]

        :raises KeyError: When the DataFrame is missing 'center_x' or 'center_y' columns
        """

        equipment = df[
            (df.center_x < self.config.INFO_MAX_X)
            &
            (df.center_y > self.config.INFO_MIN_Y)
        ]

        process = df[
            ~df.index.isin(equipment.index)
        ]

        return equipment, process

    def create_validation_pdf(
            self,
            pdf_path,
            equipment_df,
            process_df,
            output_path
    ):

        """
        Annotate a copy of the source PDF with colour-coded bounding box rectangles
        and a legend, then save it to the specified output path.

        :param self        : Instance of TextExtractor
        :type  self        : TextExtractor
        :param pdf_path    : Absolute or relative path to the source PDF file
        :type  pdf_path    : str
        :param equipment_df: DataFrame of equipment info rows, each with a 'bbox' column
        :type  equipment_df: pd.DataFrame
        :param process_df  : DataFrame of process tag rows, each with a 'bbox' column
        :type  process_df  : pd.DataFrame
        :param output_path : Destination path for the annotated validation PDF
        :type  output_path : str

        :return: None

        :raises FileNotFoundError: When the source PDF does not exist at pdf_path
        :raises OSError          : When the annotated PDF cannot be saved to output_path
        """

        doc = fitz.open(pdf_path)

        page = doc[0]

        for _, row in equipment_df.iterrows():
            b = row["bbox"]
            page.draw_rect(
                fitz.Rect(b),
                color=self.config.BBOX_COLOR_EQUIPMENT_INFO,
                width=self.config.BBOX_LINE_WIDTH
            )

        for _, row in process_df.iterrows():
            b = row["bbox"]
            page.draw_rect(
                fitz.Rect(b),
                color=self.config.BBOX_COLOR_PROCESS_TAGS,
                width=self.config.BBOX_LINE_WIDTH
            )

        page.insert_text(
            (40, 40),
            "RED = Equipment Info",
            fontsize=12,
            color=self.config.BBOX_COLOR_EQUIPMENT_INFO
        )

        page.insert_text(
            (40, 60),
            "BLUE = Process Tags",
            fontsize=12,
            color=self.config.BBOX_COLOR_PROCESS_TAGS
        )

        doc.save(output_path)
        doc.close()

        print("✓ Validation saved:", output_path)

    # ------------------------------------------------------------------
    # MERGE INSTRUMENT TAGS
    #
    # ROOT CAUSE FIX:
    #   The P&ID PDF has 90° rotation.
    #   In PDF coordinates, instrument codes and their numbers are
    #   stacked HORIZONTALLY (along the X axis), NOT vertically.
    #
    #   Example in PDF coords:
    #     TT  is at x0=298, x1=310
    #     1223 is at x0=313, x1=325   <-- to the RIGHT, same Y
    #
    #   X gap between code and number: ~3–7 pixels
    #   Y alignment: within ~6 pixels
    #
    #   OLD (broken): looked for nxt.y0 - y1 (vertical gap) → never found
    #   NEW (fixed):  looks for nxt.x0 - x1 (horizontal gap) → finds all
    # ------------------------------------------------------------------

    def merge_vertical_tags(self, df):

        """
        Merge instrument code and number spans that are horizontally adjacent in
        PDF coordinate space due to 90-degree page rotation, using a chained
        left-to-right scan with configurable gap and alignment thresholds.

        :param self: Instance of TextExtractor
        :type  self: TextExtractor
        :param df  : DataFrame with a 'text' column and bbox coordinate columns
        :type  df  : pd.DataFrame

        :return: DataFrame where horizontally adjacent instrument code and number
                 spans are combined into single rows with merged text and expanded
                 bounding boxes; unmatched rows are passed through unchanged
        :rtype : pd.DataFrame

        :raises KeyError: When the DataFrame is missing the 'text' or 'bbox' column
        """

        INSTRUMENT_CODES = self.config.INSTRUMENT_CODES

        df = self.add_bbox_columns(df)

        # Sort left → right so code always comes before its number
        work = df.sort_values(["x0", "y0"]).reset_index(drop=True)

        used = set()
        merged_rows = []

        for i in range(len(work)):

            if i in used:
                continue

            row = work.iloc[i]
            tag = str(row["text"]).strip()

            # Only start merging from a known instrument code
            if tag not in INSTRUMENT_CODES:
                merged_rows.append(row.to_dict())
                continue

            merged_text = tag
            x0 = row.x0
            y0 = row.y0
            x1 = row.x1
            y1 = row.y1

            # ---------- PASS 1: find numeric continuation to the RIGHT ----------
            for j in range(len(work)):

                if j == i or j in used:
                    continue

                nxt = work.iloc[j]
                nxt_text = str(nxt["text"]).strip()

                # Must match a number pattern (with optional letter suffix)
                # e.g. 1202, 1201A, 1201 B, 010, 036
                if not re.fullmatch(r"\d{2,5}\s*[A-Z]?", nxt_text):
                    continue

                # Must be to the RIGHT of current span
                x_gap = nxt.x0 - x1
                if x_gap < -2:       # allow tiny negative (rounding)
                    continue
                if x_gap > 15:       # max horizontal gap between code and number
                    continue

                # Must be on the same "row" (Y-aligned)
                y_diff = abs(nxt.y0 - y0)
                if y_diff > 10:
                    continue

                # MERGE
                merged_text += " " + nxt_text
                x0 = min(x0, nxt.x0)
                y0 = min(y0, nxt.y0)
                x1 = max(x1, nxt.x1)
                y1 = max(y1, nxt.y1)

                used.add(j)
                break

            merged_rows.append(
                {
                    "equipment_number": row["equipment_number"],
                    "page": row["page"],
                    "text": merged_text,
                    "bbox": (
                        round(float(x0), 4),
                        round(float(y0), 4),
                        round(float(x1), 4),
                        round(float(y1), 4)
                    )
                }
            )

        return pd.DataFrame(merged_rows)

    def merge_instrument_numbers(self, df):

        """
        Second-pass safety merge that catches any instrument code and number pairs
        missed in the first pass due to bounding box ordering differences.

        :param self: Instance of TextExtractor
        :type  self: TextExtractor
        :param df  : DataFrame with a 'text' column and bbox coordinate columns
        :type  df  : pd.DataFrame

        :return: DataFrame where any remaining unmerged instrument code and number pairs
                 are combined into single rows with merged text and expanded bounding boxes
        :rtype : pd.DataFrame

        :raises KeyError: When the DataFrame is missing the 'text' or 'bbox' column
        """

        df = self.add_bbox_columns(df)
        work = df.reset_index(drop=True)

        used = set()
        output = []

        for i, row in work.iterrows():

            if i in used:
                continue

            text1 = str(row.text).strip()

            if text1 not in self.config.INSTRUMENT_CODES:
                output.append(row.to_dict())
                continue

            best = None

            for j, row2 in work.iterrows():

                if j == i or j in used:
                    continue

                text2 = str(row2.text).strip()

                if not re.fullmatch(r"\d{2,5}\s*[A-Z]?", text2):
                    continue

                # Horizontal: number must be to the right
                x_gap = row2.x0 - row.x1
                if x_gap < -2 or x_gap > 15:
                    continue

                # Y-aligned within 10 pixels
                y_diff = abs(row2.y0 - row.y0)
                if y_diff > 10:
                    continue

                best = j
                break

            if best is not None:

                row2 = work.iloc[best]

                output.append(
                    {
                        "equipment_number": row.equipment_number,
                        "page": row.page,
                        "text": text1 + " " + str(row2.text),
                        "bbox": (
                            round(float(min(row.x0, row2.x0)), 4),
                            round(float(min(row.y0, row2.y0)), 4),
                            round(float(max(row.x1, row2.x1)), 4),
                            round(float(max(row.y1, row2.y1)), 4)
                        )
                    }
                )

                used.add(best)

            else:

                output.append(row.to_dict())

        return pd.DataFrame(output)

    def classify_tags(self, df):

        """
        Assign a semantic tag_type label to every row based on the text pattern,
        using a strict priority rule set applied in order.

        :param self: Instance of TextExtractor
        :type  self: TextExtractor
        :param df  : DataFrame containing a 'text' column of merged tag strings
        :type  df  : pd.DataFrame

        :return: Copy of the DataFrame with an additional 'tag_type' column whose
                 values are one of: Line Tag, Sensor, Instrument, Equipment,
                 Special Piping Item, Pressure Safety Valve, Corrosion Monitoring
                 System, Thermowell, Piping Component, General, or Unknown
        :rtype : pd.DataFrame

        :raises KeyError: When the DataFrame is missing the 'text' column
        """

        def classify(tag):

            t = str(tag).strip()
            if not t:
                return "Unknown"

            # ── 1. LINE TAG ──────────────────────────────────────────────
            # Format: 26"-P-101-1215-A19A-IH
            # Must have: size + inch-symbol + dash + single-letter fluid code
            # + dash + numeric section + more dash-separated segments
            if re.match(r'^\d+["\u2019\u2018]?-[A-Z]{1,3}-\d+.*-.*', t):
                return "Line Tag"

            # Split into letter-prefix and the rest
            # e.g.  "TT 1223"  → prefix="TT", rest="1223"
            #        "SP-010"   → prefix="SP", rest="010"
            #        "TW"       → prefix="TW", rest=""
            m = re.match(r'^([A-Za-z]+)([\s\-]?.*)$', t)
            if not m:
                return "General"

            prefix_raw = m.group(1).upper()
            rest = m.group(2).strip().lstrip("-").strip()
            has_digits = bool(re.search(r'\d', rest))

            # ── 2. LETTERS-ONLY / SPECIAL PREFIXES (no trailing digits) ──
            # (also applies when there ARE digits — checked first by prefix)

            # Special Piping Item
            if prefix_raw == "SP":
                return "Special Piping Item"

            # Pressure Safety Valve
            if prefix_raw == "PSV":
                return "Pressure Safety Valve"

            # Corrosion Monitoring System  (CC or CP)
            if prefix_raw in ("CC", "CP"):
                return "Corrosion Monitoring System"

            # Thermowell
            if prefix_raw == "TW":
                return "Thermowell"

            # Piping Component
            if prefix_raw == "FF":
                return "Piping Component"

            # ── 3. TWO-OR-THREE LETTER PREFIX + digits ───────────────────
            if len(prefix_raw) >= 2 and has_digits:
                last = prefix_raw[-1]
                if last == "E":
                    return "Sensor"
                if last in ("T", "I", "C", "R", "S", "V"):
                    return "Instrument"

            # ── 4. SINGLE LETTER PREFIX + digits → Equipment ─────────────
            if len(prefix_raw) == 1 and has_digits:
                equipment_map = self.config.EQUIPMENT_PREFIX_MAP
                return equipment_map.get(prefix_raw, "Equipment")

            # ── 5. FALLBACK ───────────────────────────────────────────────
            return "General"

        df = df.copy()
        df["tag_type"] = df["text"].apply(classify)

        return df

    def normalize_tags(self, df):

        """
        Normalise extracted tag text by merging any instrument code with a
        horizontally adjacent numeric continuation using a hyphen separator,
        then returning a cleaned single-token-per-row DataFrame.

        :param self: Instance of TextExtractor
        :type  self: TextExtractor
        :param df  : DataFrame with columns [equipment_number, page, text, bbox]
                     and coordinate columns x0, y0, x1
        :type  df  : pd.DataFrame

        :return: DataFrame of normalised tags where split code-number pairs are
                 rejoined with a hyphen and duplicate continuation rows are dropped
        :rtype : pd.DataFrame

        :raises KeyError: When the DataFrame is missing 'text', 'bbox', or
                          coordinate columns required for gap calculation
        """

        df = df.copy()

        df = df.sort_values(["page", "y0", "x0"]).reset_index(drop=True)

        merged = []
        skip = set()

        for i in range(len(df)):

            if i in skip:
                continue

            t1 = str(df.loc[i, "text"]).strip()
            row1 = df.loc[i]

            # default single token
            final_tag = t1

            # look ahead for number/continuation
            for j in range(i + 1, len(df)):

                if j in skip:
                    continue

                t2 = str(df.loc[j, "text"]).strip()
                row2 = df.loc[j]

                # CASE: number continuation
                if re.fullmatch(r"\d{2,5}[A-Z]?", t2):

                    x_gap = row2.x0 - row1.x1
                    y_diff = abs(row2.y0 - row1.y0)

                    if -5 <= x_gap <= 20 and y_diff <= 10:
                        final_tag = t1 + "-" + t2
                        skip.add(j)
                        break

            merged.append({
                "equipment_number": row1["equipment_number"],
                "page": row1["page"],
                "text": final_tag,
                "bbox": row1["bbox"]
            })

        return pd.DataFrame(merged)

    def process_pdf(self, path):

        """
        Run the full extraction pipeline on a single PDF file: extract text, build
        coordinates, save raw outputs, merge instrument tags, filter unwanted text,
        normalise and classify tags, remove notes, split into equipment and process
        categories, save per-PDF CSVs, and generate validation PDFs.

        :param self: Instance of TextExtractor
        :type  self: TextExtractor
        :param path: Absolute or relative path to the PDF file to process
        :type  path: str

        :return: True when the PDF has been fully processed and all outputs saved
        :rtype : bool

        :raises FileNotFoundError: When the PDF file does not exist at the given path
        :raises OSError          : When output CSV or validation PDF cannot be written
        """

        print("\nProcessing:", os.path.basename(path))

        df, eq = self.extract_text_from_pdf(path)

        # 🔥 REQUIRED: build coordinates BEFORE ANY USE
        df = self.add_bbox_columns(df)

        print("Extracted:", len(df))

        # # Create coordinates first
        # df = self.add_bbox_columns(df)

        # Save raw extracted text CSV
        raw_df = df[["equipment_number", "text", "bbox"]].copy()
        raw_csv_path = os.path.join(self.config.PER_PDF_DIR, eq + "_raw.csv")
        raw_df.to_csv(raw_csv_path, index=False)
        print(f"\n── RAW EXTRACTED TEXTS ({len(raw_df)} spans) ──")
        for _, row in raw_df.iterrows():
            print(f"  {row['text']!r}")
        print(f"── Saved raw CSV → {raw_csv_path}\n")

        # Save raw validation PDF
        raw_validation_path = os.path.join(
            self.config.RESULTS_IMAGES_DIR,
            eq + "_raw_validation_bbox.pdf"
        )
        self.create_validation_pdf(
            path,
            df[df.center_x < self.config.INFO_MAX_X],
            df[~(df.center_x < self.config.INFO_MAX_X)],
            raw_validation_path
        )

        # Merge before filtering
        df = self.merge_vertical_tags(df)

        print("After merge pass 1:", len(df))

        df = self.merge_instrument_numbers(df)

        print("After merge pass 2:", len(df))

        # Remove unwanted text
        df = df[
            ~df["text"].apply(self.is_unwanted)
        ].copy()

        df = self.normalize_tags(df)

        # CLASSIFY TAGS (NEW STEP)
        df = self.classify_tags(df)

        print("After unwanted filtering:", len(df))

        df = self.add_bbox_columns(df)

        # Notes removal
        df = self.remove_notes(df)

        print("After notes removal:", len(df))

        equipment, process = self.split_equipment(df)

        FINAL_COLS = ["equipment_number", "text", "bbox", "tag_type"]

        equipment[FINAL_COLS].rename(
            columns={"equipment_number": "file_name"}
        ).to_csv(
            os.path.join(self.config.PER_PDF_DIR, eq + "_equipment.csv"),
            index=False
        )

        process[FINAL_COLS].rename(
            columns={"equipment_number": "file_name"}
        ).to_csv(
            os.path.join(self.config.PER_PDF_DIR, eq + "_process.csv"),
            index=False
        )

        validation_path = os.path.join(
            self.config.RESULTS_IMAGES_DIR,
            eq + "_validation_bbox.pdf"
        )

        self.create_validation_pdf(
            path,
            equipment,
            process,
            validation_path
        )

        self.all_equipment_info.append(equipment)
        self.all_process_tags.append(process)

        return True

    def process_all_pdfs(self):

        """
        Discover and process all PDF files found in the configured input directory
        by calling process_pdf on each file in turn.

        :param self: Instance of TextExtractor
        :type  self: TextExtractor

        :return: True when all discovered PDFs have been processed
        :rtype : bool

        :raises OSError: When INPUT_PDF_DIR does not exist or is not accessible
        """

        files = glob.glob(
            os.path.join(self.config.INPUT_PDF_DIR, "*.pdf")
        )

        for f in files:
            self.process_pdf(f)

        return True

    def save_combined(self):

        """
        Concatenate all per-PDF equipment info and process tag DataFrames accumulated
        during processing, rename the equipment_number column to file_name, and save
        them as combined CSV files.

        :param self: Instance of TextExtractor
        :type  self: TextExtractor

        :return: None

        :raises ValueError: When all_equipment_info or all_process_tags lists are empty,
                            causing pd.concat to fail on an empty sequence
        :raises OSError   : When the output CSV files cannot be written to RESULTS_CSV_DIR
        """

        FINAL_COLS = ["equipment_number", "text", "bbox", "tag_type"]

        eq   = pd.concat(self.all_equipment_info)[FINAL_COLS].rename(
            columns={"equipment_number": "file_name"}
        )
        tags = pd.concat(self.all_process_tags)[FINAL_COLS].rename(
            columns={"equipment_number": "file_name"}
        )

        eq.to_csv(
            os.path.join(self.config.RESULTS_CSV_DIR, "equipment_info.csv"),
            index=False
        )

        tags.to_csv(
            os.path.join(self.config.RESULTS_CSV_DIR, "process_tags.csv"),
            index=False
        )

    def save_tag_type_summary(self):

        """
        Count extracted tags per tag_type across all processed PDFs and save the
        results as two summary CSV files: one aggregated across all files and one
        broken down per file.

        :param self: Instance of TextExtractor
        :type  self: TextExtractor

        :return: None

        :raises ValueError: When all_equipment_info and all_process_tags are both empty,
                            causing pd.concat to fail on an empty sequence
        :raises OSError   : When the summary CSV files cannot be written to RESULTS_CSV_DIR
        """

        if not self.all_equipment_info and not self.all_process_tags:
            print("No data to summarise.")
            return

        FINAL_COLS = ["equipment_number", "text", "bbox", "tag_type"]

        eq = pd.concat(self.all_equipment_info)[FINAL_COLS].rename(
            columns={"equipment_number": "file_name"}
        )
        tags = pd.concat(self.all_process_tags)[FINAL_COLS].rename(
            columns={"equipment_number": "file_name"}
        )

        combined = pd.concat([eq, tags], ignore_index=True)

        # ── Overall summary ──────────────────────────────────────────────
        summary = (
            combined
            .groupby("tag_type", sort=False)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        summary["percentage"] = (
                (summary["count"] / summary["count"].sum() * 100)
                .round(2)
                .astype(str) + "%"
        )

        summary_path = os.path.join(
            self.config.RESULTS_CSV_DIR, "tag_type_summary.csv"
        )
        summary.to_csv(summary_path, index=False)
        print("\n── TAG TYPE SUMMARY (all files) ──")
        print(summary.to_string(index=False))
        print(f"Saved → {summary_path}")

        # ── Per-file breakdown ───────────────────────────────────────────
        per_file = (
            combined
            .groupby(["file_name", "tag_type"], sort=False)
            .size()
            .reset_index(name="count")
            .sort_values(["file_name", "count"], ascending=[True, False])
        )

        per_file_path = os.path.join(
            self.config.RESULTS_CSV_DIR, "tag_type_per_file.csv"
        )
        per_file.to_csv(per_file_path, index=False)
        print(f"Saved → {per_file_path}\n")

    def run(self):

        """
        Entry point to execute the complete P&ID text extraction workflow:
        process all PDFs, save combined output files, and produce tag type summaries.

        :param self: Instance of TextExtractor
        :type  self: TextExtractor

        :return: True when the full pipeline completes successfully
        :rtype : bool

        :raises FileNotFoundError: When no PDFs are found in INPUT_PDF_DIR
        :raises OSError          : When combined output CSVs cannot be written
        :raises ValueError       : When no data was collected across all processed PDFs
        """

        self.process_all_pdfs()
        self.save_combined()
        self.save_tag_type_summary()

        print("\n✓ ALL DONE")

        return True