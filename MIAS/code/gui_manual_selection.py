#!/usr/bin/env python3
import csv
import json
import re
import os
import sys
import shutil
import tempfile
import subprocess
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox


class PaperSelectorApp:
    """
    GUI for:
      - Inspecting and selecting papers.
      - Inspecting, selecting, and exporting keywords.
      - Running the LLM-based keyword weighting pipeline by calling
        'keywords_weight_estimations.py' as an external script via subprocess.

    IMPORTANT (weights pipeline):
      - We do NOT import 'keywords_weight_estimations' in this process, to avoid
        LangChain/Pydantic incompatibilities with Python 3.14.
      - Instead we call the CLI:
You do not need to fix pyenv’s 3.11.9 just to run this GUI. Using the Homebrew 3.14 interpreter for this one script is perfectly fine and much simpler.
          <python_for_weights> keywords_weight_estimations.py \
              --keywords <keywords.csv> \
              --description <description.txt> \
              --instructions <instructions.yml> \
              --o <weights.csv>

      - The GUI then reads both <keywords.csv> and <weights.csv> and decorates
        the keyword table by appending (w=XX.XX) to the relevant fields.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Paper and Keyword Selector")

        # Data containers
        self.papers = []                # list of dicts (from main JSON)
        self.base_dir = ""              # directory where the main JSON lives
        self.current_paper_index = None # index into self.papers
        self.displayed_paper_indices = []  # mapping Listbox row -> index in self.papers
        self.current_json_path = None   # full path of currently loaded JSON

        # CSV cache and modification tracking for per-paper keyword CSVs
        # path -> {"fieldnames": [...], "rows": [...]}
        self.csv_data = {}
        self.modified_csvs = set()

        # Last exported combined keywords CSV (for manual exports)
        self.last_exported_keywords_csv = None

        # For last run of weighting pipeline
        self.last_weights_keywords_path = None
        self.last_weights_path = None

        # Mapping from a keyword row (paper+fields) to its weights
        # key = (
        #    paper_title (sanitized, no commas),
        #    keyword_type ("global"/"specific"),
        #    keyword, macro_area, specific_area,
        #    scientific_role, study_object, research_question
        # )
        # value = dict of weights as read from <basename>_weights.csv
        self.keyword_weight_map = {}

        # Fonts
        self.paper_font_normal = tkfont.Font(family="TkDefaultFont", size=10)

        self._build_ui()

    # -------------------------------------------------------------------------
    # UI building
    # -------------------------------------------------------------------------

    def _build_ui(self):
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        # LEFT PANEL: top buttons + paper list + doc selection buttons
        left_frame = ttk.Frame(main_pane, padding=5)
        main_pane.add(left_frame, weight=1)

        # --- top buttons ---
        btn_load = ttk.Button(left_frame, text="Load papers JSON...",
                              command=self.load_papers_json_dialog)
        btn_load.pack(anchor="w", pady=(0, 5))


        btn_save_papers = ttk.Button(left_frame, text="Save",
                                     command=self.save_paper_selection)
        btn_save_papers.pack(anchor="w", pady=(0, 5))

        # Single export entry point
        btn_export = ttk.Button(
            left_frame,
            text="Export...",
            command=self.open_export_dialog,
        )
        btn_export.pack(anchor="w", pady=(0, 5))




        # New button: run the LLM-based weighting pipeline
        btn_compute_weights = ttk.Button(
            left_frame,
            text="Compute keyword weights",
            command=self.open_weights_window,
        )
        btn_compute_weights.pack(anchor="w", pady=(0, 10))

        # --- Paper list with scrollbar ---
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        self.paper_listbox = tk.Listbox(
            list_frame,
            selectmode=tk.BROWSE,
            exportselection=False,
        )
        self.paper_listbox.config(font=self.paper_font_normal)
        yscroll = ttk.Scrollbar(
            list_frame, orient=tk.VERTICAL, command=self.paper_listbox.yview
        )
        self.paper_listbox.config(yscrollcommand=yscroll.set)

        self.paper_listbox.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")

        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        self.paper_listbox.bind("<<ListboxSelect>>", self.on_paper_select)
        self.paper_listbox.bind("<Double-Button-1>", self.on_paper_double_click)

        # --- Bottom doc selection buttons (vertical, narrow column) ---
        # --- Bottom doc selection buttons (vertical, narrow column) ---
        docs_buttons_frame = ttk.Frame(left_frame)
        docs_buttons_frame.pack(fill=tk.X, pady=(5, 0))

        # Show-all and search controls (above the selection buttons)
        btn_search_docs = ttk.Button(
            docs_buttons_frame,
            text="Search...",
            command=self.open_search_window,
        )
        btn_search_docs.pack(anchor="w", pady=(0, 5))
        
        btn_show_all_docs = ttk.Button(
            docs_buttons_frame,
            text="Show all",
            command=self.show_all_papers,
        )
        btn_show_all_docs.pack(anchor="w", pady=(0, 2))

        btn_select_all_docs = ttk.Button(
            docs_buttons_frame,
            text="Select all",
            command=self.select_all_docs,
        )
        btn_select_all_docs.pack(anchor="w", pady=(0, 2))

        btn_deselect_all_docs = ttk.Button(
            docs_buttons_frame,
            text="Deselect all",
            command=self.deselect_all_docs,
        )
        btn_deselect_all_docs.pack(anchor="w", pady=(0, 2))

        btn_invert_docs = ttk.Button(
            docs_buttons_frame,
            text="Invert selection",
            command=self.invert_docs_selection,
        )
        btn_invert_docs.pack(anchor="w", pady=(0, 0))


        # RIGHT PANEL: metadata + notebooks + keyword buttons
        right_frame = ttk.Frame(main_pane, padding=5)
        main_pane.add(right_frame, weight=3)

        # --- Metadata frame ---
        meta_frame = ttk.LabelFrame(right_frame, text="Paper metadata", padding=5)
        meta_frame.pack(fill=tk.X)

        self.meta_title_var = tk.StringVar()
        self.meta_author_var = tk.StringVar()
        self.meta_journal_var = tk.StringVar()
        self.meta_year_var = tk.StringVar()
        self.meta_doi_var = tk.StringVar()
        self.meta_type_var = tk.StringVar()
        self.meta_included_var = tk.StringVar(value="False")

        ttk.Label(meta_frame, text="Title:").grid(row=0, column=0, sticky="w")
        ttk.Label(meta_frame, textvariable=self.meta_title_var,
                  wraplength=600).grid(row=0, column=1, sticky="w")

        ttk.Label(meta_frame, text="Author:").grid(row=1, column=0, sticky="w")
        ttk.Label(meta_frame, textvariable=self.meta_author_var,
                  wraplength=600).grid(row=1, column=1, sticky="w")

        ttk.Label(meta_frame, text="Journal:").grid(row=2, column=0, sticky="w")
        ttk.Label(meta_frame, textvariable=self.meta_journal_var,
                  wraplength=600).grid(row=2, column=1, sticky="w")

        ttk.Label(meta_frame, text="Year:").grid(row=3, column=0, sticky="w")
        ttk.Label(meta_frame, textvariable=self.meta_year_var).grid(
            row=3, column=1, sticky="w"
        )

        ttk.Label(meta_frame, text="DOI:").grid(row=4, column=0, sticky="w")
        ttk.Label(meta_frame, textvariable=self.meta_doi_var,
                  wraplength=600).grid(row=4, column=1, sticky="w")

        ttk.Label(meta_frame, text="Type of study:").grid(row=5, column=0, sticky="w")
        ttk.Label(meta_frame, textvariable=self.meta_type_var,
                  wraplength=600).grid(row=5, column=1, sticky="w")

        ttk.Label(meta_frame, text="Included:").grid(row=6, column=0, sticky="w")
        ttk.Label(meta_frame, textvariable=self.meta_included_var).grid(
            row=6, column=1, sticky="w"
        )

        for i in range(2):
            meta_frame.columnconfigure(i, weight=1)

        # --- Notebook: global keywords, specific keywords, text, saved weights ---
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(5, 5))

        self.global_frame = ttk.Frame(self.notebook)
        self.specific_frame = ttk.Frame(self.notebook)
        self.text_frame = ttk.Frame(self.notebook)
        self.weights_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.global_frame, text="Global keywords")
        self.notebook.add(self.specific_frame, text="Specific keywords")
        self.notebook.add(self.text_frame, text="Text")
        self.notebook.add(self.weights_frame, text="Saved weights")

        # Global / specific keyword trees
        self.global_tree = self._create_keyword_tree(self.global_frame)
        self.specific_tree = self._create_keyword_tree(self.specific_frame)

        # Attach CSV path holders to trees
        self.global_tree.csv_path = None
        self.specific_tree.csv_path = None

        self.global_tree.bind(
            "<Double-1>", lambda e: self.on_keyword_double_click(e, is_global=True)
        )
        self.specific_tree.bind(
            "<Double-1>", lambda e: self.on_keyword_double_click(e, is_global=False)
        )

        # Text viewer (now with horizontal scrollbar; wrap disabled for horizontal scroll)
        self.text_widget = tk.Text(self.text_frame, wrap=tk.NONE)
        text_vscroll = ttk.Scrollbar(
            self.text_frame, orient=tk.VERTICAL, command=self.text_widget.yview
        )
        text_hscroll = ttk.Scrollbar(
            self.text_frame, orient=tk.HORIZONTAL, command=self.text_widget.xview
        )
        self.text_widget.config(
            yscrollcommand=text_vscroll.set,
            xscrollcommand=text_hscroll.set,
        )

        self.text_widget.grid(row=0, column=0, sticky="nsew")
        text_vscroll.grid(row=0, column=1, sticky="ns")
        text_hscroll.grid(row=1, column=0, sticky="ew")

        self.text_frame.rowconfigure(0, weight=1)
        self.text_frame.rowconfigure(1, weight=0)
        self.text_frame.columnconfigure(0, weight=1)

        # Saved weights tree (read-only, with horizontal slider)
        self.weights_tree = self._create_saved_weights_tree(self.weights_frame)

        # Bottom keyword select/deselect buttons (apply to current tab)
        kw_buttons_frame = ttk.Frame(right_frame)
        kw_buttons_frame.pack(fill=tk.X, pady=(0, 0))

        btn_select_all_kw = ttk.Button(
            kw_buttons_frame,
            text="Select all",
            command=self.select_all_keywords_current_tab,
        )
        btn_select_all_kw.pack(side=tk.LEFT, padx=(0, 5))

        btn_deselect_all_kw = ttk.Button(
            kw_buttons_frame,
            text="Deselect all",
            command=self.deselect_all_keywords_current_tab,
        )
        btn_deselect_all_kw.pack(side=tk.LEFT)

    def _create_keyword_tree(self, parent: ttk.Frame) -> ttk.Treeview:
        """
        Create a Treeview for keywords with sortable columns and a final
        'selected' column. Coloring:

        - 'selected_kw' tag: red foreground (selected keywords).
        - 'unselected_kw' tag: black foreground.

        Now includes horizontal scrollbar.
        """
        columns = (
            "keyword",
            "relevance",
            "macro_area",
            "specific_area",
            "scientific_role",
            "study_object",
            "research_question",
            "selected",
        )

        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)

        tree = ttk.Treeview(
            frame,
            columns=columns,
            show="headings",
            selectmode="browse",
        )

        for col in columns:
            tree.heading(
                col,
                text=col,
                command=lambda c=col, t=tree: self.sort_tree(t, c, False),
            )
            if col == "keyword":
                tree.column(col, width=220, anchor="w")
            elif col == "research_question":
                tree.column(col, width=220, anchor="w")
            else:
                tree.column(col, width=130, anchor="w")

        tree.tag_configure("selected_kw", foreground="red")
        tree.tag_configure("unselected_kw", foreground="black")

        yscroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        xscroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=tree.xview)
        tree.config(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)

        tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")

        frame.rowconfigure(0, weight=1)
        frame.rowconfigure(1, weight=0)
        frame.columnconfigure(0, weight=1)

        return tree

    def _create_saved_weights_tree(self, parent: ttk.Frame) -> ttk.Treeview:
        """
        Treeview for the 'Saved weights' tab.
        Shows all weighted items as "value (w=XX.XX)" for each field,
        with the whole text in blue for visual emphasis.

        Now includes horizontal scrollbar.
        """
        columns = (
            "paper_title",
            "keyword_type",
            "keyword",
            "relevance",
            "macro_area",
            "specific_area",
            "scientific_role",
            "study_object",
            "research_question",
        )

        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)

        tree = ttk.Treeview(
            frame,
            columns=columns,
            show="headings",
            selectmode="browse",
        )

        for col in columns:
            tree.heading(
                col,
                text=col,
                command=lambda c=col, t=tree: self.sort_tree(t, c, False),
            )
            if col in ("paper_title", "research_question"):
                tree.column(col, width=220, anchor="w")
            else:
                tree.column(col, width=150, anchor="w")

        # One tag for blue text (weights view)
        tree.tag_configure("weights_blue", foreground="blue")

        yscroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        xscroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=tree.xview)
        tree.config(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)

        tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")

        frame.rowconfigure(0, weight=1)
        frame.rowconfigure(1, weight=0)
        frame.columnconfigure(0, weight=1)

        # No csv_path, this tree is read-only from the weighting results
        tree.csv_path = None

        return tree

    # -------------------------------------------------------------------------
    # Loading / clearing
    # -------------------------------------------------------------------------

    def load_papers_json_dialog(self):
        json_path = filedialog.askopenfilename(
            title="Select papers structure JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not json_path:
            return
        self.load_papers_from_path(json_path)

    def load_papers_from_path(self, json_path: str):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load JSON file:\n{e}")
            return

        if not isinstance(data, list):
            messagebox.showerror("Error", "JSON must contain a list of paper entries.")
            return

        self.papers = data
        self.base_dir = os.path.dirname(json_path)
        self.current_json_path = json_path
        self.current_paper_index = None

        self.csv_data.clear()
        self.modified_csvs.clear()
        self.keyword_weight_map.clear()
        self.last_weights_keywords_path = None
        self.last_weights_path = None

        for paper in self.papers:
            sel_str = str(paper.get("selected", "False")).strip().lower()
            paper["_included"] = sel_str == "true"

        # By default show all papers
        self.displayed_paper_indices = list(range(len(self.papers)))
        self.refresh_paper_listbox()

        self.clear_metadata_and_keywords_and_text()
        # No popup "Loaded X papers..." – less annoying.


    def clear_metadata_and_keywords_and_text(self):
        self.meta_title_var.set("")
        self.meta_author_var.set("")
        self.meta_journal_var.set("")
        self.meta_year_var.set("")
        self.meta_doi_var.set("")
        self.meta_type_var.set("")
        self.meta_included_var.set("False")

        for tree in (self.global_tree, self.specific_tree):
            for item in tree.get_children():
                tree.delete(item)

        # Clear saved weights when changing dataset
        for item in self.weights_tree.get_children():
            self.weights_tree.delete(item)

        self.global_tree.csv_path = None
        self.specific_tree.csv_path = None

        self.text_widget.delete("1.0", tk.END)

    # -------------------------------------------------------------------------
    # Paper list styling / selection
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # Paper list styling / selection
    # -------------------------------------------------------------------------

    def refresh_paper_listbox(self):
        """
        Rebuild the paper Listbox from self.displayed_paper_indices.
        Each Listbox row i corresponds to self.papers[self.displayed_paper_indices[i]].
        """
        self.paper_listbox.delete(0, tk.END)

        if not self.papers:
            self.displayed_paper_indices = []
            return

        if not self.displayed_paper_indices:
            self.displayed_paper_indices = list(range(len(self.papers)))

        for _, paper_idx in enumerate(self.displayed_paper_indices):
            if paper_idx < 0 or paper_idx >= len(self.papers):
                continue
            paper = self.papers[paper_idx]
            title = paper.get("title", f"(no title {paper_idx})")
            year = paper.get("year", "N/A")
            display = f"{title} [{year}]"
            self.paper_listbox.insert(tk.END, display)
            self.update_paper_listbox_style(paper_idx)

    def update_paper_listbox_style(self, paper_idx: int):
        """
        Colour the Listbox entry corresponding to paper_idx
        (red if included, black otherwise).
        """
        if paper_idx < 0 or paper_idx >= len(self.papers):
            return

        # Map paper index -> Listbox row index
        if not self.displayed_paper_indices:
            list_idx = paper_idx
        else:
            try:
                list_idx = self.displayed_paper_indices.index(paper_idx)
            except ValueError:
                # Paper is not currently displayed
                return

        paper = self.papers[paper_idx]
        included = bool(paper.get("_included", False))
        fg = "red" if included else "black"

        try:
            self.paper_listbox.itemconfig(list_idx, fg=fg)
        except tk.TclError:
            # Listbox might not yet have that row; ignore safely
            pass

    def on_paper_select(self, event):
        if not self.papers:
            return

        selection = self.paper_listbox.curselection()
        if not selection:
            return

        list_idx = selection[0]
        if self.displayed_paper_indices:
            if list_idx >= len(self.displayed_paper_indices):
                return
            paper_idx = self.displayed_paper_indices[list_idx]
        else:
            paper_idx = list_idx

        self.current_paper_index = paper_idx
        paper = self.papers[paper_idx]

        self.meta_title_var.set(paper.get("title", ""))
        self.meta_author_var.set(paper.get("author", ""))
        self.meta_journal_var.set(paper.get("journal", ""))
        self.meta_year_var.set(paper.get("year", ""))
        self.meta_doi_var.set(paper.get("doi", ""))
        self.meta_type_var.set(paper.get("type of study", ""))

        included = bool(paper.get("_included", False))
        self.meta_included_var.set("True" if included else "False")

        for tree in (self.global_tree, self.specific_tree):
            for item in tree.get_children():
                tree.delete(item)
        self.global_tree.csv_path = None
        self.specific_tree.csv_path = None

        self.load_keywords_for_paper(paper)
        self.load_text_for_paper(paper)

    def on_paper_double_click(self, event):
        if not self.papers:
            return

        list_idx = self.paper_listbox.nearest(event.y)
        if list_idx < 0:
            return

        if self.displayed_paper_indices:
            if list_idx >= len(self.displayed_paper_indices):
                return
            paper_idx = self.displayed_paper_indices[list_idx]
        else:
            paper_idx = list_idx

        self.paper_listbox.selection_clear(0, tk.END)
        self.paper_listbox.selection_set(list_idx)
        self.paper_listbox.event_generate("<<ListboxSelect>>")

        self.toggle_paper_included(paper_idx)



    def toggle_paper_included(self, idx: int):
        paper = self.papers[idx]
        included = bool(paper.get("_included", False))
        included = not included
        paper["_included"] = included

        if self.current_paper_index == idx:
            self.meta_included_var.set("True" if included else "False")

        self.update_paper_listbox_style(idx)

    # -------------------------------------------------------------------------
    # Paper search / filtering
    # -------------------------------------------------------------------------

    def _parse_search_query(self, query: str):
        """
        Split the query into tokens, treating quoted text as a single token.

        Example:
          biopsy imaging "deep learning"
        -> ["biopsy", "imaging", "deep learning"]
        """
        pattern = r'"([^"]+)"|(\S+)'
        tokens = []
        for match in re.finditer(pattern, query):
            phrase = match.group(1) or match.group(2)
            phrase = phrase.strip()
            if phrase:
                tokens.append(phrase)
        return tokens

    def _collect_paper_search_text(self, paper_idx: int, selected_only: bool):
        """
        Build a single text blob for a paper from its
        title/metadata and its global/specific keywords.
        If selected_only is True, restrict to keyword rows whose
        'selected' column is True.
        """
        if paper_idx < 0 or paper_idx >= len(self.papers):
            return ""

        paper = self.papers[paper_idx]
        chunks = []

        # Basic metadata
        for field in ("title", "author", "journal", "year", "doi", "type of study"):
            val = paper.get(field)
            if val:
                chunks.append(str(val))

        def add_from_csv(rel_path):
            if not rel_path:
                return
            path = os.path.join(self.base_dir, rel_path)
            rows, _ = self.read_keyword_csv(path)
            for row in rows:
                if selected_only:
                    sel_str = str(row.get("selected", "False")).strip().lower()
                    if sel_str != "true":
                        continue
                for col in (
                    "keyword",
                    "research_question",
                    "macro_area",
                    "specific_area",
                    "study_object",
                    "scientific_role",
                    "relevance",
                ):
                    val = row.get(col)
                    if val:
                        chunks.append(str(val))

        global_rel_path = paper.get("keywords_global_csv") or paper.get(
            "keywords_global_file"
        )
        specific_rel_path = paper.get("keywords_specific_csv") or paper.get(
            "keywords_specific_file"
        )

        add_from_csv(global_rel_path)
        add_from_csv(specific_rel_path)

        return " ".join(chunks)

    def _paper_matches_search(self, paper_idx: int, tokens, selected_only: bool):
        """
        Return True if all tokens are found (case-insensitive) in the
        aggregated keyword / metadata text of the paper.
        """
        haystack = self._collect_paper_search_text(paper_idx, selected_only)
        if not haystack:
            return False

        haystack = haystack.lower()
        for token in tokens:
            if token.lower() not in haystack:
                return False
        return True

    def perform_search(self, query: str, selected_only: bool = False):
        """
        Perform the search and filter the paper listbox to show only matches.
        """
        if not self.papers:
            return

        query = (query or "").strip()
        if not query:
            messagebox.showinfo("Search", "Please enter search terms.")
            return

        tokens = self._parse_search_query(query)
        if not tokens:
            messagebox.showinfo("Search", "Please enter search terms.")
            return

        matching_indices = []
        for idx in range(len(self.papers)):
            if self._paper_matches_search(idx, tokens, selected_only):
                matching_indices.append(idx)

        if not matching_indices:
            messagebox.showinfo("Search", "No papers matched the search.")
            return

        self.displayed_paper_indices = matching_indices
        self.refresh_paper_listbox()
        self.current_paper_index = None

    def open_search_window(self):
        """
        Open a small search window with:
          - text entry
          - 'search selected only' checkbutton
          - 'Search' button that filters the paper list.
        """
        win = tk.Toplevel(self.root)
        win.title("Search papers")

        ttk.Label(win, text="Search terms:").grid(
            row=0, column=0, sticky="w", padx=5, pady=(5, 2)
        )

        query_var = tk.StringVar()
        entry = ttk.Entry(win, textvariable=query_var, width=60)
        entry.grid(row=0, column=1, columnspan=2, sticky="we", padx=5, pady=(5, 2))

        selected_only_var = tk.BooleanVar(value=False)
        chk = ttk.Checkbutton(
            win,
            text="Search selected keywords only",
            variable=selected_only_var,
            onvalue=True,
            offvalue=False,
        )
        chk.grid(row=1, column=0, columnspan=3, sticky="w", padx=5, pady=(0, 5))

        btn_search = ttk.Button(
            win,
            text="Search",
            command=lambda: self.perform_search(
                query_var.get(), selected_only_var.get()
            ),
        )
        btn_search.grid(row=2, column=2, sticky="e", padx=5, pady=(0, 5))

        win.columnconfigure(1, weight=1)
        entry.focus_set()

    def show_all_papers(self):
        """
        Reset the paper listbox to show all papers again.
        """
        if not self.papers:
            return
        self.displayed_paper_indices = list(range(len(self.papers)))
        self.refresh_paper_listbox()
        self.current_paper_index = None


    # -------------------------------------------------------------------------
    # Doc selection tools
    # -------------------------------------------------------------------------
    def select_all_docs(self):
        """
        Mark all *currently displayed* papers as included.
        If no filter is active (displayed_paper_indices == []),
        this falls back to all papers.
        """
        if not self.papers:
            return

        target_indices = (
            list(self.displayed_paper_indices)
            if self.displayed_paper_indices
            else list(range(len(self.papers)))
        )

        for paper_idx in target_indices:
            self.papers[paper_idx]["_included"] = True
            self.update_paper_listbox_style(paper_idx)

        if (
            self.current_paper_index is not None
            and self.current_paper_index in target_indices
        ):
            self.meta_included_var.set("True")

    def deselect_all_docs(self):
        """
        Mark all *currently displayed* papers as not included.
        If no filter is active (displayed_paper_indices == []),
        this falls back to all papers.
        """
        if not self.papers:
            return

        target_indices = (
            list(self.displayed_paper_indices)
            if self.displayed_paper_indices
            else list(range(len(self.papers)))
        )

        for paper_idx in target_indices:
            self.papers[paper_idx]["_included"] = False
            self.update_paper_listbox_style(paper_idx)

        if (
            self.current_paper_index is not None
            and self.current_paper_index in target_indices
        ):
            self.meta_included_var.set("False")

    def invert_docs_selection(self):
        """
        Invert 'included' status for all *currently displayed* papers.
        If no filter is active (displayed_paper_indices == []),
        this falls back to all papers.
        """
        if not self.papers:
            return

        target_indices = (
            list(self.displayed_paper_indices)
            if self.displayed_paper_indices
            else list(range(len(self.papers)))
        )

        for paper_idx in target_indices:
            current = bool(self.papers[paper_idx].get("_included", False))
            self.papers[paper_idx]["_included"] = not current
            self.update_paper_listbox_style(paper_idx)

        if (
            self.current_paper_index is not None
            and self.current_paper_index in target_indices
        ):
            curr_included = bool(
                self.papers[self.current_paper_index].get("_included", False)
            )
            self.meta_included_var.set("True" if curr_included else "False")

    # -------------------------------------------------------------------------
    # Keyword CSV reading
    # -------------------------------------------------------------------------

    def read_keyword_csv(self, path: str):
        if path in self.csv_data:
            data = self.csv_data[path]
            return data["rows"], data["fieldnames"]

        rows = []
        fieldnames = []
        try:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                for row in reader:
                    rows.append(row)
        except Exception as e:
            messagebox.showerror("Error", f"Could not read CSV {path}:\n{e}")
            self.csv_data[path] = {"fieldnames": fieldnames, "rows": rows}
            return rows, fieldnames

        if "selected" not in fieldnames:
            fieldnames.append("selected")
        for r in rows:
            if "selected" not in r:
                r["selected"] = "False"

        self.csv_data[path] = {"fieldnames": fieldnames, "rows": rows}
        return rows, fieldnames

    # -------------------------------------------------------------------------
    # Keyword loading and text loading
    # -------------------------------------------------------------------------

    def _decorate_keyword_row_with_weights(
        self,
        paper_title: str,
        keyword_type: str,
        row: dict,
    ):
        """
        Given a keyword row (original text values) and the paper's title/keyword_type,
        append weight information to each textual field if we have a matching
        entry in self.keyword_weight_map.

        Mapping key is:
          (paper_title_sanitized,
           keyword_type,
           keyword,
           macro_area,
           specific_area,
           scientific_role,
           study_object,
           research_question)
        """
        sanitized_title = (paper_title or "").replace(",", "")

        key = (
            sanitized_title,
            keyword_type,
            row.get("keyword", "") or "",
            row.get("macro_area", "") or "",
            row.get("specific_area", "") or "",
            row.get("scientific_role", "") or "",
            row.get("study_object", "") or "",
            row.get("research_question", "") or "",
        )

        weights = self.keyword_weight_map.get(key)

        base_vals = {
            "keyword": row.get("keyword", ""),
            "relevance": row.get("relevance", ""),
            "macro_area": row.get("macro_area", ""),
            "specific_area": row.get("specific_area", ""),
            "scientific_role": row.get("scientific_role", ""),
            "study_object": row.get("study_object", ""),
            "research_question": row.get("research_question", ""),
            "selected": row.get("selected", "False"),
        }

        if not weights:
            return (
                base_vals["keyword"],
                base_vals["relevance"],
                base_vals["macro_area"],
                base_vals["specific_area"],
                base_vals["scientific_role"],
                base_vals["study_object"],
                base_vals["research_question"],
                base_vals["selected"],
            )

        def with_weight(col_name: str) -> str:
            if col_name == "selected":
                return base_vals["selected"]
            text_val = base_vals[col_name]
            w_val = weights.get(col_name)
            if w_val is None or w_val == "":
                return text_val
            if text_val:
                return f"{text_val} (w={w_val})"
            else:
                return f"(w={w_val})"

        return (
            with_weight("keyword"),
            with_weight("relevance"),
            with_weight("macro_area"),
            with_weight("specific_area"),
            with_weight("scientific_role"),
            with_weight("study_object"),
            with_weight("research_question"),
            base_vals["selected"],
        )

    def load_keywords_for_paper(self, paper: dict):
        paper_title = paper.get("title", "")

        global_rel_path = paper.get("keywords_global_csv") or paper.get(
            "keywords_global_file"
        )
        specific_rel_path = paper.get("keywords_specific_csv") or paper.get(
            "keywords_specific_file"
        )

        if global_rel_path:
            global_path = os.path.join(self.base_dir, global_rel_path)
            rows, _ = self.read_keyword_csv(global_path)
            self.global_tree.csv_path = global_path

            for row in rows:
                vals = self._decorate_keyword_row_with_weights(
                    paper_title, "global", row
                )
                sel_flag = (
                    str(row.get("selected", "False")).strip().lower() == "true"
                )
                tag = "selected_kw" if sel_flag else "unselected_kw"
                self.global_tree.insert("", tk.END, values=vals, tags=(tag,))

        if specific_rel_path:
            specific_path = os.path.join(self.base_dir, specific_rel_path)
            rows, _ = self.read_keyword_csv(specific_path)
            self.specific_tree.csv_path = specific_path

            for row in rows:
                vals = self._decorate_keyword_row_with_weights(
                    paper_title, "specific", row
                )
                sel_flag = (
                    str(row.get("selected", "False")).strip().lower() == "true"
                )
                tag = "selected_kw" if sel_flag else "unselected_kw"
                self.specific_tree.insert("", tk.END, values=vals, tags=(tag,))

    def load_text_for_paper(self, paper: dict):
        self.text_widget.delete("1.0", tk.END)

        rel_path = paper.get("raw_text_json_filename")
        if not rel_path:
            return

        path = os.path.join(self.base_dir, rel_path)
        if not os.path.isfile(path):
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Could not read text JSON {path}:\n{e}")
            return

        text = data.get("fulltext", "")
        if not text:
            lines = []
            title = data.get("title", "")
            if title:
                lines.append("=== TITLE ===")
                lines.append(title)
                lines.append("")

            authors = data.get("authors", [])
            if authors:
                lines.append("=== AUTHORS ===")
                for a in authors:
                    if isinstance(a, dict):
                        lines.append(a.get("full_name", ""))
                    else:
                        lines.append(str(a))
                lines.append("")

            abstract = data.get("abstract", "")
            if abstract:
                lines.append("=== ABSTRACT ===")
                lines.append(abstract)
                lines.append("")

            for sec in data.get("sections", []):
                stitle = sec.get("title", "")
                stext = sec.get("text", "")
                if stitle:
                    lines.append(f"=== SECTION: {stitle} ===")
                if stext:
                    lines.append(stext)
                lines.append("")

            text = "\n".join(lines)

        self.text_widget.insert("1.0", text)

    # -------------------------------------------------------------------------
    # Keyword toggling, sync, sort
    # -------------------------------------------------------------------------

    def set_keyword_item_style(self, tree: ttk.Treeview, item_id, selected_flag: bool):
        tag = "selected_kw" if selected_flag else "unselected_kw"
        tree.item(item_id, tags=(tag,))

    def sync_tree_to_csv(self, tree: ttk.Treeview, csv_path: str):
        if csv_path not in self.csv_data:
            return
        data = self.csv_data[csv_path]
        fieldnames = list(data["fieldnames"])
        rows = []
        cols = list(tree["columns"])

        for item in tree.get_children(""):
            vals = list(tree.item(item, "values"))
            row_dict = {fn: "" for fn in fieldnames}
            for i, col in enumerate(cols):
                if i < len(vals) and col in fieldnames:
                    row_dict[col] = vals[i]
            if "selected" in fieldnames and "selected" not in row_dict:
                row_dict["selected"] = "False"
            rows.append(row_dict)

        data["rows"] = rows
        self.modified_csvs.add(csv_path)

    def on_keyword_double_click(self, event, is_global: bool):
        tree = event.widget
        item_id = tree.identify_row(event.y)
        if not item_id:
            return

        cols = list(tree["columns"])
        try:
            sel_idx = cols.index("selected")
        except ValueError:
            sel_idx = len(cols) - 1

        vals = list(tree.item(item_id, "values"))
        if len(vals) <= sel_idx:
            vals += [""] * (sel_idx + 1 - len(vals))

        current_flag = str(vals[sel_idx]).strip().lower() == "true"
        new_flag = not current_flag

        vals[sel_idx] = "True" if new_flag else "False"
        tree.item(item_id, values=tuple(vals))
        self.set_keyword_item_style(tree, item_id, new_flag)

        csv_path = getattr(tree, "csv_path", None)
        if csv_path:
            self.sync_tree_to_csv(tree, csv_path)

    def _get_current_keyword_tree(self):
        current = self.notebook.select()
        if not current:
            return None
        frame = self.notebook.nametowidget(current)
        if frame is self.global_frame:
            return self.global_tree
        if frame is self.specific_frame:
            return self.specific_tree
        # Saved weights tab is not editable/used for selection operations.
        return None

    def select_all_keywords_current_tab(self):
        tree = self._get_current_keyword_tree()
        if tree is None:
            return
        cols = list(tree["columns"])
        try:
            sel_idx = cols.index("selected")
        except ValueError:
            sel_idx = len(cols) - 1

        for item_id in tree.get_children(""):
            vals = list(tree.item(item_id, "values"))
            if len(vals) <= sel_idx:
                vals += [""] * (sel_idx + 1 - len(vals))
            vals[sel_idx] = "True"
            tree.item(item_id, values=tuple(vals))
            self.set_keyword_item_style(tree, item_id, True)

        csv_path = getattr(tree, "csv_path", None)
        if csv_path:
            self.sync_tree_to_csv(tree, csv_path)

    def deselect_all_keywords_current_tab(self):
        tree = self._get_current_keyword_tree()
        if tree is None:
            return
        cols = list(tree["columns"])
        try:
            sel_idx = cols.index("selected")
        except ValueError:
            sel_idx = len(cols) - 1

        for item_id in tree.get_children(""):
            vals = list(tree.item(item_id, "values"))
            if len(vals) <= sel_idx:
                vals += [""] * (sel_idx + 1 - len(vals))
            vals[sel_idx] = "False"
            tree.item(item_id, values=tuple(vals))
            self.set_keyword_item_style(tree, item_id, False)

        csv_path = getattr(tree, "csv_path", None)
        if csv_path:
            self.sync_tree_to_csv(tree, csv_path)

    def sort_tree(self, tree: ttk.Treeview, col: str, reverse: bool):
        col_index = list(tree["columns"]).index(col)
        data = []
        for item in tree.get_children(""):
            vals = tree.item(item, "values")
            val = vals[col_index] if col_index < len(vals) else ""
            try:
                key = float(val)
            except (ValueError, TypeError):
                key = val.lower() if isinstance(val, str) else val
            data.append((key, item))

        data.sort(key=lambda x: x[0], reverse=reverse)
        for idx, (_, item) in enumerate(data):
            tree.move(item, "", idx)

        tree.heading(
            col,
            command=lambda c=col, t=tree: self.sort_tree(t, c, not reverse),
        )

        csv_path = getattr(tree, "csv_path", None)
        if csv_path:
            self.sync_tree_to_csv(tree, csv_path)

    # -------------------------------------------------------------------------
    # Save / export papers and keywords
    # -------------------------------------------------------------------------

    def save_paper_selection(self):
        """
        Save in-place:
        - overwrite the currently loaded JSON file (self.current_json_path)
        - write back any modified keyword CSVs
        No file dialog is opened.
        """
        if not self.papers:
            messagebox.showwarning("No data", "Load a papers JSON first.")
            return

        if not self.current_json_path:
            messagebox.showerror(
                "Error",
                "No source JSON file to overwrite. Use 'Load papers JSON...' first.",
            )
            return

        # propagate _included back to 'selected'
        for paper in self.papers:
            included = bool(paper.get("_included", False))
            paper["selected"] = "True" if included else "False"

        out_path = self.current_json_path

        # write JSON
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(self.papers, f, indent=2, ensure_ascii=False)
        except Exception as e:
            messagebox.showerror("Error", f"Could not write JSON file:\n{e}")
            return

        # write any modified per-paper keyword CSVs
        for path in list(self.modified_csvs):
            data = self.csv_data.get(path)
            if not data:
                continue

            fieldnames = list(data["fieldnames"])
            rows = data["rows"]

            if "selected" not in fieldnames:
                fieldnames.append("selected")
            for r in rows:
                if "selected" not in r:
                    r["selected"] = "False"

            try:
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
            except Exception as e:
                messagebox.showerror("Error", f"Could not write CSV file {path}:\n{e}")
                continue

            self.modified_csvs.discard(path)

        messagebox.showinfo(
            "Saved",
            f"Papers JSON and modified keyword CSVs have been saved to:\n{out_path}",
        )

    def export_json_as(self):
        """
        Export the current papers JSON to a new file (Save As),
        and write any modified keyword CSVs.
        This is equivalent to the previous 'Save' behaviour.
        """
        if not self.papers:
            messagebox.showwarning("No data", "Load a papers JSON first.")
            return

        # propagate _included back to 'selected'
        for paper in self.papers:
            included = bool(paper.get("_included", False))
            paper["selected"] = "True" if included else "False"

        out_path = filedialog.asksaveasfilename(
            title="Export updated papers JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not out_path:
            return

        # write JSON
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(self.papers, f, indent=2, ensure_ascii=False)
        except Exception as e:
            messagebox.showerror("Error", f"Could not write JSON file:\n{e}")
            return

        # write any modified per-paper keyword CSVs
        for path in list(self.modified_csvs):
            data = self.csv_data.get(path)
            if not data:
                continue

            fieldnames = list(data["fieldnames"])
            rows = data["rows"]

            if "selected" not in fieldnames:
                fieldnames.append("selected")
            for r in rows:
                if "selected" not in r:
                    r["selected"] = "False"

            try:
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
            except Exception as e:
                messagebox.showerror("Error", f"Could not write CSV file {path}:\n{e}")
                continue

            self.modified_csvs.discard(path)

        messagebox.showinfo(
            "Exported",
            f"Papers JSON and modified keyword CSVs have been exported to:\n{out_path}",
        )

    def open_export_dialog(self):
        """
        Open a modal export dialog with radio buttons:
          - Export JSON (Save As)
          - Export selected docs
          - Export global keywords
          - Export specific keywords
        """
        if not self.papers:
            messagebox.showwarning("No data", "Load a papers JSON first.")
            return

        win = tk.Toplevel(self.root)
        win.title("Export")
        win.transient(self.root)
        win.grab_set()  # make it modal

        ttk.Label(win, text="Select what to export:").grid(
            row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(10, 5)
        )

        choice_var = tk.StringVar(value="json")

        options = [
            ("Export JSON (Save As)", "json"),
            ("Export selected docs", "docs"),
            ("Export global keywords", "global_kw"),
            ("Export specific keywords", "specific_kw"),
        ]

        for i, (label, value) in enumerate(options, start=1):
            ttk.Radiobutton(
                win,
                text=label,
                variable=choice_var,
                value=value,
            ).grid(row=i, column=0, columnspan=2, sticky="w", padx=10, pady=2)

        def do_export():
            choice = choice_var.get()
            if choice == "json":
                self.export_json_as()
            elif choice == "docs":
                self.export_selected_docs()
            elif choice == "global_kw":
                self.export_keywords(scope="global")
            elif choice == "specific_kw":
                self.export_keywords(scope="specific")
            win.destroy()

        btn_export = ttk.Button(win, text="Export", command=do_export)
        btn_export.grid(
            row=len(options) + 1, column=0, sticky="e", padx=10, pady=(10, 10)
        )

        btn_cancel = ttk.Button(win, text="Cancel", command=win.destroy)
        btn_cancel.grid(
            row=len(options) + 1, column=1, sticky="w", padx=5, pady=(10, 10)
        )

        win.columnconfigure(0, weight=1)


    def export_selected_docs(self):
        if not self.papers:
            messagebox.showwarning("No data", "Load a papers JSON first.")
            return

        selected_papers = [p for p in self.papers if bool(p.get("_included", False))]
        if not selected_papers:
            messagebox.showinfo("Export selected docs", "No documents are selected.")
            return

        out_path = filedialog.asksaveasfilename(
            title="Export selected documents JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not out_path:
            return

        for p in selected_papers:
            p["selected"] = "True"

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(selected_papers, f, indent=2, ensure_ascii=False)
        except Exception as e:
            messagebox.showerror("Error", f"Could not write JSON file:\n{e}")
            return

        messagebox.showinfo(
            "Export selected docs",
            f"Exported {len(selected_papers)} selected documents to:\n{out_path}",
        )

    def _collect_selected_keywords_rows(self, scope: str = "all"):
        """
        Collect all SELECTED keywords (selected == True) from all papers
        into a list of rows with fields:

          paper_title, keyword_type, keyword, relevance, macro_area,
          specific_area, scientific_role, study_object, research_question

        'scope' can be "all", "global", or "specific".
        """
        rows_out = []

        if not self.papers:
            return rows_out

        for paper in self.papers:
            title = paper.get("title", "")
            sanitized_title = title.replace(",", "")

            def collect_from_csv(rel_path, keyword_type):
                if not rel_path:
                    return
                path = os.path.join(self.base_dir, rel_path)

                if path in self.csv_data:
                    rows = self.csv_data[path]["rows"]
                else:
                    rows, _ = self.read_keyword_csv(path)

                for r in rows:
                    sel = str(r.get("selected", "False")).strip().lower() == "true"
                    if not sel:
                        continue
                    rows_out.append(
                        {
                            "paper_title": sanitized_title,
                            "keyword_type": keyword_type,
                            "keyword": r.get("keyword", ""),
                            "relevance": r.get("relevance", ""),
                            "macro_area": r.get("macro_area", ""),
                            "specific_area": r.get("specific_area", ""),
                            "scientific_role": r.get("scientific_role", ""),
                            "study_object": r.get("study_object", ""),
                            "research_question": r.get("research_question", ""),
                        }
                    )

            global_rel_path = paper.get("keywords_global_csv") or paper.get(
                "keywords_global_file"
            )
            specific_rel_path = paper.get("keywords_specific_csv") or paper.get(
                "keywords_specific_file"
            )

            if scope in ("global", "all"):
                collect_from_csv(global_rel_path, "global")
            if scope in ("specific", "all"):
                collect_from_csv(specific_rel_path, "specific")

        return rows_out

    def export_keywords(self, scope: str = "all"):
        """
        Export selected keywords from all papers into a single CSV with columns:
          paper_title, keyword_type, keyword, relevance, macro_area,
          specific_area, scientific_role, study_object, research_question

        This CSV is exactly what the weighting pipeline expects as --keywords.
        """
        if not self.papers:
            messagebox.showwarning("No data", "Load a papers JSON first.")
            return

        out_path = filedialog.asksaveasfilename(
            title="Export selected keywords",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not out_path:
            return

        fieldnames = [
            "paper_title",
            "keyword_type",
            "keyword",
            "relevance",
            "macro_area",
            "specific_area",
            "scientific_role",
            "study_object",
            "research_question",
        ]

        export_rows = self._collect_selected_keywords_rows(scope=scope)

        try:
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(export_rows)
        except Exception as e:
            messagebox.showerror(
                "Error", f"Could not write exported keyword file:\n{e}"
            )
            return

        self.last_exported_keywords_csv = out_path

        messagebox.showinfo(
            "Exported",
            f"Exported {len(export_rows)} selected keywords to:\n{out_path}",
        )

    # -------------------------------------------------------------------------
    # Weighting pipeline dialog (subprocess-based, using selected keywords)
    # -------------------------------------------------------------------------

    def open_weights_window(self):
        """
        Open a child window to configure and run the LLM-based keyword
        weighting pipeline.

        IMPORTANT: We no longer ask for an external keywords CSV.
        Instead, the keywords are built from the CURRENT SELECTIONS in the GUI:

          - All papers, both global and specific keyword tables.
          - Only rows with selected == True are included.

        The GUI maps to CLI arguments as follows:

          --keywords     -> auto-generated combined CSV of selected keywords:
                            <out_dir>/<basename>_keywords.csv
          --description  -> temporary text file created from description Text
          --instructions -> YAML path chosen via 'Browse...' button (instr_var)
          --o            -> <out_dir>/<basename>_weights.csv

        where out_dir is self.base_dir (directory of the loaded papers JSON)
        or, if that is not available, the current working directory.
        """

        win = tk.Toplevel(self.root)
        win.title("Compute keyword weights")

        desc_label = ttk.Label(
            win,
            text=(
                "Description (information need):\n"
                "Type or paste here the description of what you are looking for.\n"
                "The weighting pipeline will run on the CURRENTLY SELECTED keywords\n"
                "(global + specific) across all papers in this GUI."
            ),
            wraplength=600,
            justify="left",
        )
        desc_label.pack(anchor="w", padx=5, pady=(5, 2))

        desc_frame = ttk.Frame(win)
        desc_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        desc_text = tk.Text(desc_frame, height=8, wrap=tk.WORD)
        desc_scroll = ttk.Scrollbar(
            desc_frame, orient=tk.VERTICAL, command=desc_text.yview
        )
        desc_text.config(yscrollcommand=desc_scroll.set)

        desc_text.grid(row=0, column=0, sticky="nsew")
        desc_scroll.grid(row=0, column=1, sticky="ns")
        desc_frame.rowconfigure(0, weight=1)
        desc_frame.columnconfigure(0, weight=1)

        # Instructions YAML
        instr_frame = ttk.Frame(win)
        instr_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        ttk.Label(instr_frame, text="Instructions YAML:").grid(
            row=0, column=0, sticky="w"
        )
        instr_var = tk.StringVar(value="")
        instr_entry = ttk.Entry(instr_frame, textvariable=instr_var, width=70)
        instr_entry.grid(row=0, column=1, sticky="we", padx=(5, 5))
        instr_frame.columnconfigure(1, weight=1)

        def browse_instr():
            path = filedialog.askopenfilename(
                title="Select instructions YAML",
                filetypes=[("YAML files", "*.yml *.yaml"), ("All files", "*.*")],
            )
            if path:
                instr_var.set(path)

        instr_btn = ttk.Button(instr_frame, text="Browse...", command=browse_instr)
        instr_btn.grid(row=0, column=2, padx=(0, 0))

        # Python interpreter for weighting
        py_frame = ttk.Frame(win)
        py_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        ttk.Label(
            py_frame,
            text=(
                "Python for weighting (interpreter that can run langchain etc.):\n"
                "Default tries your pyenv 3.11 at ~/.pyenv/versions/3.11.9/bin/python.\n"
                "If that does not exist, it falls back to 'python'."
            ),
            justify="left",
        ).grid(row=0, column=0, columnspan=3, sticky="w")

        default_py = os.path.expanduser("~/.pyenv/versions/3.11.9/bin/python")
        if not os.path.isfile(default_py):
            default_py = "python"

        ttk.Label(py_frame, text="Python executable:").grid(
            row=1, column=0, sticky="w", pady=(2, 0)
        )
        python_var = tk.StringVar(value=default_py)
        python_entry = ttk.Entry(py_frame, textvariable=python_var, width=70)
        python_entry.grid(row=1, column=1, sticky="we", padx=(5, 5))
        py_frame.columnconfigure(1, weight=1)

        # Basename for output files
        base_frame = ttk.Frame(win)
        base_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        ttk.Label(
            base_frame,
            text=(
                "Output basename (e.g. 'my_analysis'):\n"
                "This will create in the project directory:\n"
                "  <basename>_keywords.csv  (combined selected keywords)\n"
                "  <basename>_weights.csv   (computed weights)\n"
                "The 'Saved weights' tab will display these results as\n"
                "text entries like 'imaging (w=80.00)' in blue."
            ),
            justify="left",
        ).grid(row=0, column=0, columnspan=3, sticky="w")

        ttk.Label(base_frame, text="Basename:").grid(
            row=1, column=0, sticky="w", pady=(2, 0)
        )
        base_var = tk.StringVar(value="weights")
        base_entry = ttk.Entry(base_frame, textvariable=base_var, width=40)
        base_entry.grid(row=1, column=1, sticky="w", padx=(5, 5))

        # Status label
        status_var = tk.StringVar(value="")
        status_label = ttk.Label(win, textvariable=status_var, foreground="blue")
        status_label.pack(fill=tk.X, padx=5, pady=(0, 5))

        # Run + Close buttons
        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        def run_pipeline():
            """
            1. Collect CURRENTLY SELECTED keywords from the GUI
               (global + specific, all papers) into a combined CSV:
                   <out_dir>/<basename>_keywords.csv

            2. Read DESCRIPTION from desc_text -> write to a temp file.

            3. Get instructions YAML path, Python executable, and basename.

            4. Compute:
                 weights_path  = <out_dir>/<basename>_weights.csv
                 keywords_path = <out_dir>/<basename>_keywords.csv

            5. Call external CLI:

                 <python_for_weights> keywords_weight_estimations.py \
                     --keywords <keywords_path> \
                     --description <tmp_desc> \
                     --instructions <instructions.yml> \
                     --o <weights_path>

            6. Load weights into self.keyword_weight_map, refresh current paper,
               and populate the 'Saved weights' tab.
            """
            description = desc_text.get("1.0", "end").strip()
            instr_path = instr_var.get().strip()
            basename = base_var.get().strip()
            python_exe = python_var.get().strip() or "python"

            if not self.papers:
                messagebox.showerror(
                    "Error", "Please load a papers JSON and select keywords first."
                )
                return
            if not description:
                messagebox.showerror(
                    "Error", "Please provide a description (information need)."
                )
                return
            if not instr_path or not os.path.isfile(instr_path):
                messagebox.showerror(
                    "Error", "Please select a valid instructions YAML file."
                )
                return
            if not basename:
                messagebox.showerror("Error", "Please provide a non-empty basename.")
                return

            # Collect selected keywords from GUI
            combined_rows = self._collect_selected_keywords_rows(scope="all")
            if not combined_rows:
                messagebox.showerror(
                    "Error",
                    "No keywords are selected.\n\n"
                    "Please select some keywords (global/specific) in the main GUI "
                    "before running the weighting pipeline.",
                )
                return

            out_dir = self.base_dir or os.getcwd()
            os.makedirs(out_dir, exist_ok=True)

            keywords_path = os.path.join(out_dir, f"{basename}_keywords.csv")
            weights_path = os.path.join(out_dir, f"{basename}_weights.csv")

            fieldnames = [
                "paper_title",
                "keyword_type",
                "keyword",
                "relevance",
                "macro_area",
                "specific_area",
                "scientific_role",
                "study_object",
                "research_question",
            ]

            try:
                with open(keywords_path, "w", newline="", encoding="utf-8") as f_kw:
                    writer = csv.DictWriter(f_kw, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(combined_rows)
            except Exception as e:
                messagebox.showerror(
                    "Error",
                    f"Could not write combined keywords CSV:\n{keywords_path}\n\n{e}",
                )
                return

            status_var.set("Running weighting pipeline... this may take a while.")
            win.update_idletasks()

            tmp_desc_fd, tmp_desc_path = tempfile.mkstemp(
                suffix=".txt", prefix="desc_", text=True
            )
            os.close(tmp_desc_fd)

            try:
                with open(tmp_desc_path, "w", encoding="utf-8") as f_desc:
                    f_desc.write(description)

                # Path to keywords_weight_estimations.py (same folder as this GUI)
                script_dir = os.path.dirname(os.path.abspath(__file__))
                kw_script_path = os.path.join(script_dir, "keywords_weight_estimations.py")

                if not os.path.isfile(kw_script_path):
                    raise RuntimeError(
                        f"keywords_weight_estimations.py not found at:\n{kw_script_path}"
                    )

                cmd = [
                    python_exe,
                    kw_script_path,
                    "--keywords",
                    keywords_path,
                    "--description",
                    tmp_desc_path,
                    "--instructions",
                    instr_path,
                    "--o",
                    weights_path,
                ]

                subprocess.run(cmd, check=True)

                # Load weights into memory for GUI decoration
                self._load_weights_into_map(keywords_path, weights_path)

                # Remember last paths for reference
                self.last_weights_keywords_path = keywords_path
                self.last_weights_path = weights_path

                # Refresh current paper view to show weights in keyword tabs
                if self.current_paper_index is not None:
                    idx = self.current_paper_index
                    self.paper_listbox.selection_clear(0, tk.END)
                    self.paper_listbox.selection_set(idx)
                    self.on_paper_select(None)

                # Populate the "Saved weights" tab
                self.populate_saved_weights_tab(keywords_path)

                status_var.set(
                    f"Done.\nWeights: {weights_path}\nKeywords: {keywords_path}"
                )
                messagebox.showinfo(
                    "Weights computed",
                    f"Keyword weights computed.\n\n"
                    f"Weights CSV:\n  {weights_path}\n"
                    f"Keywords CSV:\n  {keywords_path}",
                )
            except subprocess.CalledProcessError as e:
                status_var.set("")
                messagebox.showerror(
                    "Error",
                    "Weighting pipeline failed.\n\n"
                    f"Command:\n{' '.join(cmd)}\n\n"
                    f"Exit code: {e.returncode}",
                )
            except Exception as e:
                status_var.set("")
                messagebox.showerror(
                    "Error", f"Weighting pipeline failed:\n{e}"
                )
            finally:
                try:
                    os.remove(tmp_desc_path)
                except OSError:
                    pass

        btn_run = ttk.Button(btn_frame, text="Run", command=run_pipeline)
        btn_run.pack(side=tk.LEFT, padx=(0, 5))

        btn_close = ttk.Button(btn_frame, text="Close", command=win.destroy)
        btn_close.pack(side=tk.LEFT)

    def _load_weights_into_map(self, keywords_path: str, weights_path: str):
        """
        Load the original keywords CSV and the weights CSV, align rows by index,
        and populate self.keyword_weight_map.

        Both CSVs are assumed to have the same header and number of rows.
        The weights CSV has paper_title preserved and all other columns filled
        with numeric scores as strings.
        """
        with open(keywords_path, newline="", encoding="utf-8") as f_kw:
            kw_reader = csv.DictReader(f_kw)
            kw_fieldnames = kw_reader.fieldnames or []
            kw_rows = list(kw_reader)

        with open(weights_path, newline="", encoding="utf-8") as f_w:
            w_reader = csv.DictReader(f_w)
            w_fieldnames = w_reader.fieldnames or []
            w_rows = list(w_reader)

        if len(kw_rows) != len(w_rows):
            messagebox.showwarning(
                "Weights mismatch",
                "Warning: keywords CSV and weights CSV have different numbers of rows.\n"
                "Weights will not be fully applied.",
            )
        n = min(len(kw_rows), len(w_rows))

        self.keyword_weight_map.clear()

        for orig, w in zip(kw_rows[:n], w_rows[:n]):
            paper_title = orig.get("paper_title", "")
            keyword_type = orig.get("keyword_type", "")
            key = (
                paper_title,
                keyword_type,
                orig.get("keyword", "") or "",
                orig.get("macro_area", "") or "",
                orig.get("specific_area", "") or "",
                orig.get("scientific_role", "") or "",
                orig.get("study_object", "") or "",
                orig.get("research_question", "") or "",
            )

            weight_entry = {}
            for col in w_fieldnames:
                if col == "paper_title":
                    continue
                weight_entry[col] = w.get(col, "")

            self.keyword_weight_map[key] = weight_entry

    def populate_saved_weights_tab(self, keywords_path: str):
        """
        Fill the 'Saved weights' tab with a table of all rows and their
        associated weights, in the form 'value (w=XX.XX)' for each field.
        The entire text is coloured blue using the 'weights_blue' tag.
        """
        # Clear previous content
        for item in self.weights_tree.get_children():
            self.weights_tree.delete(item)

        try:
            with open(keywords_path, newline="", encoding="utf-8") as f_kw:
                reader = csv.DictReader(f_kw)
                rows = list(reader)
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Could not read keywords CSV for saved weights:\n{keywords_path}\n\n{e}",
            )
            return

        for orig in rows:
            paper_title = orig.get("paper_title", "")
            keyword_type = orig.get("keyword_type", "")

            key = (
                paper_title,
                keyword_type,
                orig.get("keyword", "") or "",
                orig.get("macro_area", "") or "",
                orig.get("specific_area", "") or "",
                orig.get("scientific_role", "") or "",
                orig.get("study_object", "") or "",
                orig.get("research_question", "") or "",
            )

            weights = self.keyword_weight_map.get(key, {})

            def join_w(col_name: str, base_val: str):
                if col_name == "paper_title":
                    return base_val
                w_val = weights.get(col_name)
                if not w_val:
                    return base_val
                if base_val:
                    return f"{base_val} (w={w_val})"
                else:
                    return f"(w={w_val})"

            row_values = [
                join_w("paper_title", paper_title),
                join_w("keyword_type", keyword_type),
                join_w("keyword", orig.get("keyword", "") or ""),
                join_w("relevance", orig.get("relevance", "") or ""),
                join_w("macro_area", orig.get("macro_area", "") or ""),
                join_w("specific_area", orig.get("specific_area", "") or ""),
                join_w("scientific_role", orig.get("scientific_role", "") or ""),
                join_w("study_object", orig.get("study_object", "") or ""),
                join_w(
                    "research_question", orig.get("research_question", "") or ""
                ),
            ]

            self.weights_tree.insert(
                "", tk.END, values=row_values, tags=("weights_blue",)
            )

    # -------------------------------------------------------------------------
    # main entry point
    # -------------------------------------------------------------------------


def main():
    root = tk.Tk()
    app = PaperSelectorApp(root)
    root.geometry("1300x750")

    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        if os.path.isfile(json_path):
            app.load_papers_from_path(json_path)

    root.mainloop()


if __name__ == "__main__":
    main()
