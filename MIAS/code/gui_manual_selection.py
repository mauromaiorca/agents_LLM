#!/usr/bin/env python3
import json
import csv
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkfont


class PaperSelectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Paper and Keyword Selector")

        # Data containers
        self.papers = []                # list of paper dicts (from main JSON)
        self.base_dir = ""              # directory where main JSON lives
        self.current_paper_index = None # index of currently selected paper in self.papers

        # CSV data cache and modification tracking
        # path -> {"fieldnames": [...], "rows": [...]}
        self.csv_data = {}
        self.modified_csvs = set()

        # Fonts and styles (only used where Tk supports them)
        self.paper_font_normal = tkfont.Font(family="TkDefaultFont", size=10)

        # Build UI
        self._build_ui()

    # ---------- UI BUILDING ----------

    def _build_ui(self):
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        # LEFT PANEL: buttons + paper list
        left_frame = ttk.Frame(main_pane, padding=5)
        main_pane.add(left_frame, weight=1)

        # Top buttons
        btn_load = ttk.Button(left_frame, text="Load papers JSON...", command=self.load_papers_json_dialog)
        btn_load.pack(anchor="w", pady=(0, 5))

        btn_save_papers = ttk.Button(left_frame, text="Save", command=self.save_paper_selection)
        btn_save_papers.pack(anchor="w", pady=(0, 5))

        btn_export_docs = ttk.Button(left_frame, text="Export selected docs", command=self.export_selected_docs)
        btn_export_docs.pack(anchor="w", pady=(0, 5))

        # Export keyword buttons
        btn_export_all = ttk.Button(left_frame, text="Export all keywords",
                                    command=lambda: self.export_keywords(scope="all"))
        btn_export_all.pack(anchor="w", pady=(0, 2))

        btn_export_global = ttk.Button(left_frame, text="Export global keywords",
                                       command=lambda: self.export_keywords(scope="global"))
        btn_export_global.pack(anchor="w", pady=(0, 2))

        btn_export_specific = ttk.Button(left_frame, text="Export specific keywords",
                                         command=lambda: self.export_keywords(scope="specific"))
        btn_export_specific.pack(anchor="w", pady=(0, 5))

        # Paper list with scrollbar
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        self.paper_listbox = tk.Listbox(
            list_frame,
            selectmode=tk.BROWSE,
            exportselection=False
        )
        self.paper_listbox.config(font=self.paper_font_normal)
        yscroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.paper_listbox.yview)
        self.paper_listbox.config(yscrollcommand=yscroll.set)

        self.paper_listbox.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")

        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        self.paper_listbox.bind("<<ListboxSelect>>", self.on_paper_select)
        self.paper_listbox.bind("<Double-Button-1>", self.on_paper_double_click)

        # Bottom buttons for docs select/deselect/invert, stacked vertically
        docs_buttons_frame = ttk.Frame(left_frame)
        docs_buttons_frame.pack(fill=tk.X, pady=(5, 0))

        btn_select_all_docs = ttk.Button(docs_buttons_frame, text="Select all",
                                         command=self.select_all_docs)
        btn_select_all_docs.pack(anchor="w", pady=(0, 2))

        btn_deselect_all_docs = ttk.Button(docs_buttons_frame, text="Deselect all",
                                           command=self.deselect_all_docs)
        btn_deselect_all_docs.pack(anchor="w", pady=(0, 2))

        btn_invert_docs = ttk.Button(docs_buttons_frame, text="Invert selection",
                                     command=self.invert_docs_selection)
        btn_invert_docs.pack(anchor="w", pady=(0, 0))

        # RIGHT PANEL: metadata + tabs + keyword controls at bottom
        right_frame = ttk.Frame(main_pane, padding=5)
        main_pane.add(right_frame, weight=3)

        # Metadata frame
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
        ttk.Label(meta_frame, textvariable=self.meta_title_var, wraplength=600).grid(row=0, column=1, sticky="w")

        ttk.Label(meta_frame, text="Author:").grid(row=1, column=0, sticky="w")
        ttk.Label(meta_frame, textvariable=self.meta_author_var, wraplength=600).grid(row=1, column=1, sticky="w")

        ttk.Label(meta_frame, text="Journal:").grid(row=2, column=0, sticky="w")
        ttk.Label(meta_frame, textvariable=self.meta_journal_var, wraplength=600).grid(row=2, column=1, sticky="w")

        ttk.Label(meta_frame, text="Year:").grid(row=3, column=0, sticky="w")
        ttk.Label(meta_frame, textvariable=self.meta_year_var).grid(row=3, column=1, sticky="w")

        ttk.Label(meta_frame, text="DOI:").grid(row=4, column=0, sticky="w")
        ttk.Label(meta_frame, textvariable=self.meta_doi_var, wraplength=600).grid(row=4, column=1, sticky="w")

        ttk.Label(meta_frame, text="Type of study:").grid(row=5, column=0, sticky="w")
        ttk.Label(meta_frame, textvariable=self.meta_type_var, wraplength=600).grid(row=5, column=1, sticky="w")

        ttk.Label(meta_frame, text="Included:").grid(row=6, column=0, sticky="w")
        ttk.Label(meta_frame, textvariable=self.meta_included_var).grid(row=6, column=1, sticky="w")

        for i in range(2):
            meta_frame.columnconfigure(i, weight=1)

        # Tabs: global keywords, specific keywords, text
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(5, 5))

        self.global_frame = ttk.Frame(self.notebook)
        self.specific_frame = ttk.Frame(self.notebook)
        self.text_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.global_frame, text="Global keywords")
        self.notebook.add(self.specific_frame, text="Specific keywords")
        self.notebook.add(self.text_frame, text="Text")

        # Keyword trees with scrollbars
        self.global_tree = self._create_keyword_tree(self.global_frame)
        self.specific_tree = self._create_keyword_tree(self.specific_frame)

        # Bind double-click on keywords
        self.global_tree.bind("<Double-1>", lambda e: self.on_keyword_double_click(e, is_global=True))
        self.specific_tree.bind("<Double-1>", lambda e: self.on_keyword_double_click(e, is_global=False))

        # Text viewer with scrollbar
        self.text_widget = tk.Text(self.text_frame, wrap=tk.WORD)
        text_scroll = ttk.Scrollbar(self.text_frame, orient=tk.VERTICAL, command=self.text_widget.yview)
        self.text_widget.config(yscrollcommand=text_scroll.set)

        self.text_widget.grid(row=0, column=0, sticky="nsew")
        text_scroll.grid(row=0, column=1, sticky="ns")

        self.text_frame.rowconfigure(0, weight=1)
        self.text_frame.columnconfigure(0, weight=1)

        # Keyword select/deselect buttons (per current tab) at bottom
        kw_buttons_frame = ttk.Frame(right_frame)
        kw_buttons_frame.pack(fill=tk.X, pady=(0, 0))

        btn_select_all_kw = ttk.Button(kw_buttons_frame, text="Select all",
                                       command=self.select_all_keywords_current_tab)
        btn_select_all_kw.pack(side=tk.LEFT, padx=(0, 5))

        btn_deselect_all_kw = ttk.Button(kw_buttons_frame, text="Deselect all",
                                         command=self.deselect_all_keywords_current_tab)
        btn_deselect_all_kw.pack(side=tk.LEFT)

        # Attach csv_path attributes to trees (filled later)
        self.global_tree.csv_path = None
        self.specific_tree.csv_path = None

    def _create_keyword_tree(self, parent):
        # Add 'selected' column at the end
        columns = (
            "keyword",
            "relevance",
            "macro_area",
            "specific_area",
            "scientific_role",
            "study_object",
            "research_question",
            "selected"
        )

        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)

        # selectmode=browse to avoid weird multi-selection effects
        tree = ttk.Treeview(frame, columns=columns, show="headings", selectmode="browse")

        for col in columns:
            tree.heading(col, text=col,
                         command=lambda c=col, t=tree: self.sort_tree(t, c, False))
            if col == "keyword":
                tree.column(col, width=220, anchor="w")
            elif col == "research_question":
                tree.column(col, width=220, anchor="w")
            else:
                tree.column(col, width=130, anchor="w")

        # Tag styles: colour only (no fonts here to avoid TclError)
        tree.tag_configure("selected_kw", foreground="red")
        tree.tag_configure("unselected_kw", foreground="black")

        yscroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.config(yscrollcommand=yscroll.set)

        tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")

        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        return tree

    # ---------- DATA LOADING ----------

    def load_papers_json_dialog(self):
        json_path = filedialog.askopenfilename(
            title="Select papers structure JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not json_path:
            return
        self.load_papers_from_path(json_path)

    def load_papers_from_path(self, json_path):
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
        self.current_paper_index = None

        # Reset CSV caches for new dataset
        self.csv_data.clear()
        self.modified_csvs.clear()

        # Normalise 'selected' for papers into a bool in '_included'
        for paper in self.papers:
            sel_str = str(paper.get("selected", "False")).strip().lower()
            paper["_included"] = (sel_str == "true")

        # Populate listbox
        self.paper_listbox.delete(0, tk.END)
        for idx, paper in enumerate(self.papers):
            title = paper.get("title", f"(no title {idx})")
            year = paper.get("year", "N/A")
            display = f"{title} [{year}]"
            self.paper_listbox.insert(tk.END, display)
            self.update_paper_listbox_style(idx)

        # Clear detail views
        self.clear_metadata_and_keywords_and_text()

        messagebox.showinfo("Loaded", f"Loaded {len(self.papers)} papers from JSON.")

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

        self.global_tree.csv_path = None
        self.specific_tree.csv_path = None

        self.text_widget.delete("1.0", tk.END)

    # ---------- PAPER LIST STYLING ----------

    def update_paper_listbox_style(self, idx):
        if idx < 0 or idx >= len(self.papers):
            return
        paper = self.papers[idx]
        included = bool(paper.get("_included", False))
        if included:
            self.paper_listbox.itemconfig(idx, fg="red")
        else:
            self.paper_listbox.itemconfig(idx, fg="black")

    # ---------- PAPER SELECTION & TOGGLING ----------

    def on_paper_select(self, event):
        if not self.papers:
            return

        selection = self.paper_listbox.curselection()
        if not selection:
            return

        idx = selection[0]
        self.current_paper_index = idx
        paper = self.papers[idx]

        # Update metadata
        self.meta_title_var.set(paper.get("title", ""))
        self.meta_author_var.set(paper.get("author", ""))
        self.meta_journal_var.set(paper.get("journal", ""))
        self.meta_year_var.set(paper.get("year", ""))
        self.meta_doi_var.set(paper.get("doi", ""))
        self.meta_type_var.set(paper.get("type of study", ""))

        included = bool(paper.get("_included", False))
        self.meta_included_var.set("True" if included else "False")

        # Reload keywords and text
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

        idx = self.paper_listbox.nearest(event.y)
        if idx < 0 or idx >= len(self.papers):
            return

        # Ensure listbox selection follows double-click
        self.paper_listbox.selection_clear(0, tk.END)
        self.paper_listbox.selection_set(idx)
        self.paper_listbox.event_generate("<<ListboxSelect>>")

        # Toggle inclusion
        self.toggle_paper_included(idx)

    def toggle_paper_included(self, idx):
        paper = self.papers[idx]
        included = bool(paper.get("_included", False))
        included = not included
        paper["_included"] = included

        if self.current_paper_index == idx:
            self.meta_included_var.set("True" if included else "False")

        self.update_paper_listbox_style(idx)

    # ---------- DOCS SELECT/DESELECT/INVERT ----------

    def select_all_docs(self):
        for idx, paper in enumerate(self.papers):
            paper["_included"] = True
            self.update_paper_listbox_style(idx)
        if self.current_paper_index is not None:
            self.meta_included_var.set("True")

    def deselect_all_docs(self):
        for idx, paper in enumerate(self.papers):
            paper["_included"] = False
            self.update_paper_listbox_style(idx)
        if self.current_paper_index is not None:
            self.meta_included_var.set("False")

    def invert_docs_selection(self):
        for idx, paper in enumerate(self.papers):
            current = bool(paper.get("_included", False))
            paper["_included"] = not current
            self.update_paper_listbox_style(idx)
        if self.current_paper_index is not None:
            curr_included = bool(self.papers[self.current_paper_index].get("_included", False))
            self.meta_included_var.set("True" if curr_included else "False")

    # ---------- KEYWORDS LOADING ----------

    def read_keyword_csv(self, path):
        """
        Read a keyword CSV at 'path' into self.csv_data, add 'selected'
        field if missing, and return (rows, fieldnames).
        If already loaded, return cached values.
        """
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

    def load_keywords_for_paper(self, paper):
        # Determine relative paths
        global_rel_path = paper.get("keywords_global_csv") or paper.get("keywords_global_file")
        specific_rel_path = paper.get("keywords_specific_csv") or paper.get("keywords_specific_file")

        # Global
        if global_rel_path:
            global_path = os.path.join(self.base_dir, global_rel_path)
            rows, _ = self.read_keyword_csv(global_path)
            self.global_tree.csv_path = global_path

            for row in rows:
                vals = (
                    row.get("keyword", ""),
                    row.get("relevance", ""),
                    row.get("macro_area", ""),
                    row.get("specific_area", ""),
                    row.get("scientific_role", ""),
                    row.get("study_object", ""),
                    row.get("research_question", ""),
                    row.get("selected", "False"),
                )
                sel_flag = str(row.get("selected", "False")).strip().lower() == "true"
                tag = "selected_kw" if sel_flag else "unselected_kw"
                self.global_tree.insert("", tk.END, values=vals, tags=(tag,))

        # Specific
        if specific_rel_path:
            specific_path = os.path.join(self.base_dir, specific_rel_path)
            rows, _ = self.read_keyword_csv(specific_path)
            self.specific_tree.csv_path = specific_path

            for row in rows:
                vals = (
                    row.get("keyword", ""),
                    row.get("relevance", ""),
                    row.get("macro_area", ""),
                    row.get("specific_area", ""),
                    row.get("scientific_role", ""),
                    row.get("study_object", ""),
                    row.get("research_question", ""),
                    row.get("selected", "False"),
                )
                sel_flag = str(row.get("selected", "False")).strip().lower() == "true"
                tag = "selected_kw" if sel_flag else "unselected_kw"
                self.specific_tree.insert("", tk.END, values=vals, tags=(tag,))

    # ---------- TEXT LOADING ----------

    def load_text_for_paper(self, paper):
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

        # Prefer 'fulltext' if available
        text = data.get("fulltext", "")
        if not text:
            # Fallback: build something readable from title/authors/sections
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

    # ---------- KEYWORD TOGGLING ----------

    def set_keyword_item_style(self, tree, item_id, selected_flag):
        tag = "selected_kw" if selected_flag else "unselected_kw"
        tree.item(item_id, tags=(tag,))

    def sync_tree_to_csv(self, tree, csv_path):
        """Synchronise tree order and values back to csv_data[csv_path]."""
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

    def on_keyword_double_click(self, event, is_global):
        tree = event.widget
        item_id = tree.identify_row(event.y)
        if not item_id:
            return

        # Read current 'selected' value from the row and toggle it
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

        # Update colour tag
        self.set_keyword_item_style(tree, item_id, new_flag)

        # Sync to csv_data
        csv_path = getattr(tree, "csv_path", None)
        if csv_path:
            self.sync_tree_to_csv(tree, csv_path)

    # ---------- KEYWORDS SELECT/DESELECT ALL (CURRENT TAB) ----------

    def _get_current_keyword_tree(self):
        current = self.notebook.select()
        if not current:
            return None
        if self.notebook.nametowidget(current) is self.global_frame:
            return self.global_tree
        if self.notebook.nametowidget(current) is self.specific_frame:
            return self.specific_tree
        return None  # text tab or unknown

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

    # ---------- TREE SORTING ----------

    def sort_tree(self, tree, col, reverse):
        """Sort treeview by given column; toggle reverse on each call."""
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

        tree.heading(col, command=lambda c=col, t=tree: self.sort_tree(t, c, not reverse))

        # Sync new order back to CSV
        csv_path = getattr(tree, "csv_path", None)
        if csv_path:
            self.sync_tree_to_csv(tree, csv_path)

    # ---------- SAVE / EXPORT ----------

    def save_paper_selection(self):
        if not self.papers:
            messagebox.showwarning("No data", "Load a papers JSON first.")
            return

        # Update 'selected' field in papers from '_included'
        for paper in self.papers:
            included = bool(paper.get("_included", False))
            paper["selected"] = "True" if included else "False"

        out_path = filedialog.asksaveasfilename(
            title="Save updated papers JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not out_path:
            return

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(self.papers, f, indent=2, ensure_ascii=False)
        except Exception as e:
            messagebox.showerror("Error", f"Could not write JSON file:\n{e}")
            return

        # Save only modified CSVs
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

        messagebox.showinfo("Saved", "Papers JSON and modified keyword CSVs have been saved.")

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
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not out_path:
            return

        # Ensure 'selected' field in exported entries
        for p in selected_papers:
            p["selected"] = "True"

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(selected_papers, f, indent=2, ensure_ascii=False)
        except Exception as e:
            messagebox.showerror("Error", f"Could not write JSON file:\n{e}")
            return

        messagebox.showinfo("Export selected docs", f"Exported {len(selected_papers)} selected documents to:\n{out_path}")

    def export_keywords(self, scope="all"):
        if not self.papers:
            messagebox.showwarning("No data", "Load a papers JSON first.")
            return

        out_path = filedialog.asksaveasfilename(
            title="Export selected keywords",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not out_path:
            return

        fieldnames = [
            "paper_title",
            "keyword_type",  # "global" or "specific"
            "keyword",
            "relevance",
            "macro_area",
            "specific_area",
            "scientific_role",
            "study_object",
            "research_question",
        ]

        export_rows = []

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
                    export_rows.append({
                        "paper_title": sanitized_title,
                        "keyword_type": keyword_type,
                        "keyword": r.get("keyword", ""),
                        "relevance": r.get("relevance", ""),
                        "macro_area": r.get("macro_area", ""),
                        "specific_area": r.get("specific_area", ""),
                        "scientific_role": r.get("scientific_role", ""),
                        "study_object": r.get("study_object", ""),
                        "research_question": r.get("research_question", ""),
                    })

            global_rel_path = paper.get("keywords_global_csv") or paper.get("keywords_global_file")
            specific_rel_path = paper.get("keywords_specific_csv") or paper.get("keywords_specific_file")

            if scope in ("global", "all"):
                collect_from_csv(global_rel_path, "global")
            if scope in ("specific", "all"):
                collect_from_csv(specific_rel_path, "specific")

        try:
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(export_rows)
        except Exception as e:
            messagebox.showerror("Error", f"Could not write exported keyword file:\n{e}")
            return

        messagebox.showinfo(
            "Exported",
            f"Exported {len(export_rows)} selected keywords to:\n{out_path}"
        )


def main():
    root = tk.Tk()
    app = PaperSelectorApp(root)
    root.geometry("1300x750")

    # If a JSON path is passed as CLI argument, load it automatically
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        if os.path.isfile(json_path):
            app.load_papers_from_path(json_path)

    root.mainloop()


if __name__ == "__main__":
    main()

