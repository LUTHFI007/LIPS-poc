import json
from datetime import datetime
from pathlib import Path

# Load secrets from a local .env (APP_DB_URI, ADMIN_EMAILS, MLFLOW_TRACKING_URI)
# BEFORE importing our modules, so they see the values at import time. On a
# deployed Space there is no .env — the platform injects these as real env vars
# and load_dotenv() is a harmless no-op.
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import streamlit as st

from lips_poc.data_hub import search_datasets
from lips_poc.model_hub import (
    search_models, validate_upload_inputs, upload_model, upload_new_version,
    list_versions, next_version, get_version_metadata, version_for_revision,
)
from lips_poc import tracking
from lips_poc import auth
from evaluation_runner import run_evaluation, extract_scores

_ROOT = Path(__file__).parent
_LOGO_PATH = _ROOT / "img" / "Logo.png"
_MLFLOW_SPACE_URL = "https://huggingface.co/spaces/lips-poc/lips-mlflow"

# scoreboard.json is retired: the leaderboard now reads and writes solely via the
# MLflow tracking store (lips_poc/tracking.py). The old file is kept in the repo
# as a frozen artifact in case we ever want to revive local persistence.
_DOCS_DIR = _ROOT / "docs"

# The exact benchmark config the scoreboard evaluates against (see
# lips_poc/scoreboard.py). Served for download so users train against a
# byte-identical file — never copied/edited, so train-time == eval-time.
_BENCH_CONFIG_PATH = _ROOT / "configurations/powergrid/benchmarks/benchmark.ini"


def _doc_bytes(filename: str) -> bytes:
    """Read a file from docs/ as bytes for st.download_button. Passing bytes
    (not an open handle) makes browsers honour the exact download file_name."""
    return (_DOCS_DIR / filename).read_bytes()

with (_ROOT / "dataset_registry.json").open() as f:
    DATASET_REGISTRY = json.load(f)


@st.cache_data(ttl=300)
def _fetch_datasets() -> list[dict]:
    return search_datasets("")


@st.cache_data(ttl=300)
def _fetch_models() -> list[dict]:
    return search_models("")


@st.cache_data(ttl=300)
def _fetch_versions(repo_id: str) -> list[dict]:
    """A model's versions (newest first), enriched with author/parent read from
    the HF card. Cached so the expandable list stays responsive."""
    rows = []
    for v in list_versions(repo_id):
        meta = get_version_metadata(repo_id, v["version"])
        rows.append({**v, "author": meta["author"], "parent_version": meta["parent_version"]})
    return rows


# Clear stale selections on every new browser session
if "initialized" not in st.session_state:
    st.session_state.selected_dataset  = None
    st.session_state.selected_model    = None
    st.session_state.selected_version  = None
    st.session_state.selected_revision = None
    st.session_state.initialized       = True

# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="LIPS Power Grid Benchmark", layout="wide")


def _render_logo() -> None:
    """The System X logo, if present."""
    if _LOGO_PATH.exists():
        st.image(str(_LOGO_PATH), width=110)


def _status_line(ok: bool, msg: str) -> None:
    """A live ✓ / ✗ line under a register field, coloured green or red."""
    icon, colour = ("✅", "green") if ok else ("❌", "red")
    st.markdown(f":{colour}[{icon} {msg}]")


def _scoreboard_selection() -> list:
    """Indices of the scoreboard rows the admin has checked, read from the
    st.dataframe selection state (persisted in session_state under 'sb_table').
    Read at the top of the tab so the Delete button — rendered above the table —
    reflects the current selection."""
    state = st.session_state.get("sb_table")
    if not state:
        return []
    sel = state.get("selection") if isinstance(state, dict) else getattr(state, "selection", None)
    if sel is None:
        return []
    rows = sel.get("rows") if isinstance(sel, dict) else getattr(sel, "rows", None)
    return list(rows or [])


def _render_login_form() -> None:
    """Inline login form shown on the page for logged-out visitors (Register
    stays a modal in the header). On success, signs in and reruns."""
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")
    if submitted:
        found = auth.authenticate(email, password)
        if found:
            st.session_state.user = found
            st.rerun()
        else:
            st.error("Invalid email or password.")


def _render_register_form() -> None:
    """Inline registration form with live validation, shown on the page when the
    header's Register toggle is active. Fields are plain inputs (not st.form) so
    each keystroke reruns and refreshes the ✓/✗ status; the Create button stays
    disabled until every check passes. auth.register() is still the real guard."""

    # Email — format check, then uniqueness (only query the DB once the format
    # is valid, so we don't hit Neon on every partial keystroke).
    r_email = st.text_input("Email", key="reg_email")
    email_ok = False
    if r_email:
        err = auth.email_format_error(r_email)
        if err:
            _status_line(False, err)
        elif auth.email_taken(r_email):
            _status_line(False, "This email is already registered.")
        else:
            _status_line(True, "Email available.")
            email_ok = True

    # Username — same pattern, with the rules shown up front.
    r_username = st.text_input("Username", key="reg_username")
    st.caption(f"Allowed: {auth.USERNAME_RULE_TEXT}")
    username_ok = False
    if r_username:
        err = auth.username_format_error(r_username)
        if err:
            _status_line(False, err)
        elif auth.username_taken(r_username):
            _status_line(False, "This username is already taken.")
        else:
            _status_line(True, "Username available.")
            username_ok = True

    # Password — length and match.
    r_password = st.text_input("Password", type="password", key="reg_pw")
    r_confirm = st.text_input("Confirm password", type="password", key="reg_pw2")
    pw_len_ok = len(r_password) >= auth.MIN_PASSWORD_LEN
    pw_match = bool(r_password) and r_password == r_confirm
    if r_password:
        _status_line(pw_len_ok, f"At least {auth.MIN_PASSWORD_LEN} characters.")
    if r_confirm:
        _status_line(pw_match, "Passwords match." if pw_match else "Passwords do not match.")

    all_ok = email_ok and username_ok and pw_len_ok and pw_match
    if st.button("Create account", type="primary", disabled=not all_ok):
        try:
            st.session_state.user = auth.register(r_email, r_username, r_password)
            st.rerun()
        except ValueError as e:
            st.error(str(e))


def _render_header() -> None:
    """Top header bar shown in every state: logo + title on the left; on the
    right either a Register button (logged out — login is an inline form on the
    page) or the signed-in identity + Logout (logged in)."""
    logo_col, title_col, action_col = st.columns([2, 4, 3], vertical_alignment="center")
    with logo_col:
        _render_logo()
    with title_col:
        st.markdown("#### LIPS Benchmark System")
    with action_col:
        if "user" in st.session_state:
            u = st.session_state.user
            badge = " · admin" if u["role"] == "admin" else ""
            info_col, btn_col = st.columns([3, 1], vertical_alignment="center")
            info_col.markdown(f"Signed in as **{u['username']}**{badge}")
            if btn_col.button("Logout"):
                del st.session_state.user
                st.rerun()
        else:
            # Toggle between the two inline forms: the button always offers the
            # OTHER view than the one currently shown. A left spacer column pushes
            # it to the right edge of the header; both use the primary colour.
            _sp, tgl_col = st.columns([2, 1])
            if st.session_state.get("auth_view", "login") == "login":
                if tgl_col.button("Register", type="primary", use_container_width=True):
                    st.session_state.auth_view = "register"
                    st.rerun()
            else:
                if tgl_col.button("Login", type="primary", use_container_width=True):
                    st.session_state.auth_view = "login"
                    st.rerun()
    st.divider()


_render_header()

# ── Auth gate: the tabs only render for signed-in users ───────────────────────
if "user" not in st.session_state:
    _left, form_col, _right = st.columns([1, 2, 1])
    with form_col:
        if st.session_state.get("auth_view", "login") == "register":
            st.subheader("Register")
            st.caption("Already have an account? Click **Login** to access your account.")
            _render_register_form()
        else:
            st.subheader("Login")
            st.caption("New here? Click **Register** to create an account.")
            _render_login_form()
    st.stop()

user = st.session_state.user
is_admin = user["role"] == "admin"

# Note: author is no longer a free-text field. It's bound to the signed-in user's
# username (stamped on the HF card at upload) and read back from HF metadata at
# evaluation time, so every run is attributed to the version's real creator
# regardless of who clicks Evaluate.

_tab_labels = ["Data Hub", "Model Hub", "Scoreboard"] + (["Admin"] if is_admin else [])
_tabs = st.tabs(_tab_labels)
tab_data, tab_model, tab_scoreboard = _tabs[0], _tabs[1], _tabs[2]
tab_admin = _tabs[3] if is_admin else None

# ── Data Hub ──────────────────────────────────────────────────────────────────

with tab_data:
    st.subheader("Data Hub")
    st.caption("Click a row to select it.")
    datasets = _fetch_datasets()
    if not datasets:
        st.warning("No datasets found on HuggingFace (lips-poc org).")
    else:
        ds_df = pd.DataFrame(datasets)
        ds_event = st.dataframe(
            ds_df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="ds_table",
        )
        selected_ds_rows = ds_event.selection.rows
        if selected_ds_rows:
            st.session_state.selected_dataset = ds_df.iloc[selected_ds_rows[0]]["Dataset ID"]
            st.success(f"Selected: **{st.session_state.selected_dataset}**")

# ── Model Hub ─────────────────────────────────────────────────────────────────

with tab_model:
    st.subheader("Model Hub")
    st.caption("Click a model row, then choose its version from the dropdown below.")
    models = _fetch_models()
    if not models:
        st.warning("No models found on HuggingFace (lips-poc org).")
    else:
        # One row per model, with a Versions column listing its tags.
        versions_by_repo = {}
        rows = []
        for m in models:
            repo_id = m["Model ID"]
            versions = _fetch_versions(repo_id)
            versions_by_repo[repo_id] = versions
            rows.append({
                "Model ID":      repo_id,
                "Versions":      ", ".join(v["version"] for v in reversed(versions)) or "(none)",
                "Last Modified": m.get("Last Modified", ""),
                "URL":           m.get("URL", ""),
            })
        m_df = pd.DataFrame(rows)
        m_event = st.dataframe(
            m_df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="m_table",
        )

        sel_rows = m_event.selection.rows
        if sel_rows:
            repo_id  = m_df.iloc[sel_rows[0]]["Model ID"]
            versions = versions_by_repo.get(repo_id, [])
            if not versions:
                st.warning("This model has no version tags yet.")
            else:
                ver_names = [v["version"] for v in versions]  # newest first
                chosen = st.selectbox("Version", ver_names, key=f"ver::{repo_id}")
                vinfo = next(v for v in versions if v["version"] == chosen)
                st.session_state.selected_model    = repo_id
                st.session_state.selected_version  = chosen
                st.session_state.selected_revision = vinfo["revision"]
                detail = f"author `{vinfo['author']}` · commit `{vinfo['revision'][:8]}`"
                if vinfo["parent_version"]:
                    detail += f" · parent {vinfo['parent_version']}"
                st.success(
                    f"Selected **{repo_id.split('/')[-1]} @ {chosen}** — {detail}"
                )

    st.divider()
    st.subheader("Build Your Own Model")

    with st.expander("How to build and submit a model — read the instructions carefully!"):
        st.markdown("""
        ### Step-by-step

        **Step 0 — Set up your environment (install LIPS)**
        You train your model locally against the real LIPS library, so install it
        first. LIPS needs **Python ≥ 3.6** (3.10 recommended). Create an isolated
        environment, then install LIPS and its dependencies from source — the same
        steps as the [LIPS repo](https://github.com/IRT-SystemX/LIPS):

        Create and activate a conda env (recommended):
        ```
        conda create -n venv_lips python=3.10
        conda activate venv_lips
        ```
        *Or* use a plain virtualenv instead:
        ```
        pip3 install -U virtualenv
        python3 -m virtualenv venv_lips
        source venv_lips/bin/activate
        ```
        Then install LIPS. From PyPI:
        ```
        pip install "lips-benchmark[recommended]"
        ```
        *Or* from source (use `-e` for an editable checkout you can modify):
        ```
        git clone https://github.com/IRT-SystemX/LIPS.git
        cd LIPS
        pip3 install -U .[recommended]   # or:  pip3 install -e .[recommended]
        ```
        The `[recommended]` extra pulls in TensorFlow, PyTorch, Grid2Op and the
        other dependencies the templates need. Run all the steps below from inside
        this activated environment.

        **Step 1 — Download the template for your framework**
        Use the buttons below. The template is a Python file with one class you
        fill in. Everything else (data loading, preprocessing, training loop,
        save/restore) is handled by LIPS automatically.

        **Step 2 — Fill in `build_model()`**
        Open the template and replace the example architecture in `build_model()`
        with your own. Read the comments — they explain exactly what variables are
        available and what you must assign.

        **Step 3 — Download and adjust the `.ini` config**
        The `.ini` file controls your model's hyperparameters (layer sizes, learning
        rate, etc.). Download the one for your framework below, rename it (any name
        except `benchmark.ini`), and update the values to match what
        you put in `build_model()`.

        **Step 3b — Download the benchmark config**
        Also download `benchmark.ini` (button below). This is the fixed
        benchmark every model is scored against. Point `BENCH_CONFIG` in the template
        at it and train against it **as-is** — do not edit or rename it. It is *not*
        part of your ZIP; it stays on your machine for training only. Training against
        a different benchmark config produces invalid scores.

        **Step 4 — Train your model**
        Run the training script at the bottom of the template file:
        ```
        python augmented_simulator.py
        ```
        After training, LIPS saves your model into a folder. The exact files depend
        on the framework (TensorFlow vs PyTorch). `config.json` and `losses.json` are
        always written; `scaler_params.json` appears only if you trained with a scaler
        (optional). The weights filename and `metadata.json` differ by framework — see
        the two layouts below. ZIP whatever files the folder actually contains.

        **Step 5 — Assemble the ZIP**
        Copy two files into your saved model folder:
        - `augmented_simulator.py` (your filled-in template)
        - `simulator.ini` (your adjusted config file)

        Then ZIP the entire folder. Match the layout for your framework:
        """)

        st.markdown("**TensorFlow** (`custom_tf`):")
        st.code("""
your-model-name.zip
├── weights.h5                ← from sim.save()
├── config.json               ← from sim.save()
├── losses.json               ← from sim.save()
├── scaler_params.json        ← from sim.save() (only if you used a scaler — optional)
├── simulator.ini          ← you provide (downloaded and adjusted below)
└── augmented_simulator.py ← you provide (downloaded and filled in below)
        """)

        st.markdown("**PyTorch** (`custom_torch`):")
        st.code("""
your-model-name.zip
├── model_last.pt             ← from sim.save()  (weights — note: .pt, not .h5)
├── config.json               ← from sim.save()
├── losses.json               ← from sim.save()
├── metadata.json             ← from sim.save()
├── scaler_params.json        ← from sim.save() (only if you used a scaler — optional)
├── simulator.ini          ← you provide (downloaded and adjusted below)
└── augmented_simulator.py ← you provide (downloaded and filled in below)
        """)

        st.markdown("""
        **Step 6 — Upload**
        Use the upload form below. Select `custom_tf` or `custom_torch` as the
        model type. The system validates your ZIP before uploading.
        """)

    col_tf, col_torch = st.columns(2)

    with col_tf:
        st.markdown("**TensorFlow**")
        st.download_button(
            "Download TF template",
            data=_doc_bytes("custom_model_template.py"),
            file_name="augmented_simulator.py",
            mime="application/octet-stream",
        )
        st.download_button(
            "Download TF simulator config",
            data=_doc_bytes("tf_fc.ini"),
            file_name="simulator.ini",
            mime="application/octet-stream",
        )

    with col_torch:
        st.markdown("**PyTorch**")
        st.download_button(
            "Download PyTorch template",
            data=_doc_bytes("custom_model_template_torch.py"),
            file_name="augmented_simulator.py",
            mime="application/octet-stream",
        )
        st.download_button(
            "Download PyTorch simulator config",
            data=_doc_bytes("torch_fc.ini"),
            file_name="simulator.ini",
            mime="application/octet-stream",
        )

    st.markdown("**Benchmark config (required for training — same for both frameworks)**")
    st.download_button(
        "Download benchmark config",
        data=_BENCH_CONFIG_PATH.read_bytes(),
        file_name="benchmark.ini",
        mime="application/octet-stream",
    )
    st.caption(
        "Train against this exact file — point `BENCH_CONFIG` in the template at it. "
        "Do **not** edit or rename it, and do **not** include it in your ZIP. It is the "
        "fixed config the scoreboard evaluates every model against; a mismatch produces "
        "invalid scores."
    )

    st.subheader("Upload Your Model")

    upload_mode = st.radio(
        "Upload mode",
        ["New model", "New version of existing model"],
        horizontal=True,
    )
    is_new_version = upload_mode == "New version of existing model"

    # Author is bound to the signed-in user — you can only upload under your own
    # username, so it is shown read-only, not entered.
    author = user["username"]
    st.text_input("Uploading as", value=author, disabled=True)

    if is_new_version:
        existing = [m["Model ID"].split("/")[-1] for m in (_fetch_models() or [])]
        repo_name = st.selectbox("Existing model", existing) if existing else None
        if repo_name:
            st.caption(
                f"Will publish **lips-poc/{repo_name}** as "
                f"**{next_version(f'lips-poc/{repo_name}')}**."
            )
    else:
        repo_name = st.text_input("Repository name", placeholder="my-model")
        if repo_name:
            st.caption(f"Will be uploaded as: `lips-poc/{repo_name}` (Version **v0**)")

    model_type = st.selectbox(
        "Model type",
        ["tf_fc", "tf_leapnet", "torch_fc", "custom_tf", "custom_torch"],
    )
    if is_new_version:
        st.caption("Use the same model type as the existing model.")

    if model_type in ("custom_tf", "custom_torch"):
        framework = "TensorFlow" if model_type == "custom_tf" else "PyTorch"
        st.info(
            f"Custom {framework} model. Your ZIP must contain "
            "`augmented_simulator.py` and a `.ini` config file. "
            "See the instructions above."
        )

    description = st.text_area("Description (optional)")

    zip_bytes = None
    if model_type != "dc_approximation":
        uploaded = st.file_uploader("Model ZIP", type=["zip"])
        if uploaded:
            zip_bytes = uploaded.read()

    # On the rerun that follows a successful upload, show the success message and
    # skip validation for this render. Otherwise the just-created repo would be
    # re-detected by validate_upload_inputs and falsely reported as "already exists".
    just_uploaded = st.session_state.pop("upload_success", None)
    if just_uploaded:
        st.success(f"Uploaded successfully as `{just_uploaded}`.")
        errors = []
    else:
        errors = validate_upload_inputs(
            model_type, repo_name or "", zip_bytes, new_version=is_new_version
        )
        for err in errors:
            st.error(err)

    if st.button("Confirm Upload", type="primary", disabled=bool(errors)):
        with st.spinner("Uploading and validating…"):
            try:
                if is_new_version:
                    repo_id, new_tag = upload_new_version(
                        repo_name, model_type, zip_bytes, description, author=author,
                    )
                    st.session_state.upload_success = f"{repo_id} ({new_tag})"
                else:
                    repo_id = upload_model(
                        repo_name, model_type, zip_bytes, description, author=author,
                    )
                    st.session_state.upload_success = repo_id
                st.cache_data.clear()
                st.session_state.selected_model = repo_id
                st.rerun()
            except Exception as e:
                st.error(f"Upload failed: {e}")

# ── Scoreboard ────────────────────────────────────────────────────────────────

with tab_scoreboard:
    st.subheader("Scoreboard")

    sel_ds = st.session_state.get("selected_dataset")
    sel_m   = st.session_state.get("selected_model") or None
    sel_ver = st.session_state.get("selected_version")
    sel_rev = st.session_state.get("selected_revision")

    col1, col2 = st.columns(2)
    col1.metric("Selected Dataset", sel_ds or "None")
    col2.metric(
        "Selected Model",
        f"{sel_m.split('/')[-1]} @ {sel_ver}" if sel_m else "None",
    )

    # Fetch the leaderboard once — used by both the delete action (needs run_ids)
    # and the table below. Each row carries an internal 'run_id'.
    sb_rows = tracking.fetch_leaderboard()
    run_ids = [r.get("run_id") for r in sb_rows]

    # Action row: Evaluate on the left; for admins, Delete on the right at the
    # same level. Delete stays disabled until at least one run is checked.
    c_eval, _sp, c_del, c_mlflow = st.columns([2, 5, 2, 2], vertical_alignment="center")
    evaluate_clicked = c_eval.button("Evaluate", type="primary")
    if is_admin:
        sel_idx = _scoreboard_selection()
        del_label = f"🗑 Delete ({len(sel_idx)})" if sel_idx else "🗑 Delete"
        if c_del.button(del_label, disabled=not sel_idx, use_container_width=True):
            ids = [run_ids[i] for i in sel_idx if i < len(run_ids) and run_ids[i]]
            try:
                n = tracking.hard_delete_runs(ids)
                st.session_state.pop("sb_table", None)  # clear stale selection
                st.success(f"Permanently deleted {n} run(s).")
                st.rerun()
            except Exception as e:
                st.error(f"Delete failed: {e}")
        # Opens the hosted MLflow tracking Space (admin only — normal users
        # never see this button).
        c_mlflow.link_button("MLFlow", _MLFLOW_SPACE_URL, use_container_width=True)

    if evaluate_clicked:
        if not sel_ds:
            st.error("Please select a dataset in the Data Hub tab first.")
        elif not sel_m:
            st.error("Please select a model in the Model Hub tab first.")
        else:
            ds_key = sel_ds if sel_ds in DATASET_REGISTRY else sel_ds.split("/")[-1]
            if ds_key not in DATASET_REGISTRY:
                st.error(f"'{ds_key}' not found in dataset_registry.json.")
            else:
                with st.spinner("Downloading model and running evaluation — this may take a few minutes…"):
                    try:
                        results, hf_revision, model_config = run_evaluation(
                            dataset_info=DATASET_REGISTRY[ds_key],
                            model_repo_id=sel_m,
                            revision=sel_rev,
                        )
                        scores = extract_scores(results)
                    except Exception as e:
                        import traceback
                        st.error(f"Evaluation failed: {e}\n\n```\n{traceback.format_exc()}\n```")
                        st.stop()

                new_row = {
                    "Model":     sel_m.split("/")[-1],
                    "Dataset":   ds_key,
                    "Benchmark": DATASET_REGISTRY[ds_key]["benchmark_name"],
                    **scores,
                    "Timestamp": datetime.now().isoformat(timespec="seconds"),
                }
                # Tracking plane: log this evaluation to MLflow, linked to the
                # exact HF commit via hf_revision. version + author + parent come
                # from the HF version metadata (the card at this commit), so the
                # run is attributed to whoever created the version — not whoever
                # clicked Evaluate. Observability only — never blocks the result.
                meta    = get_version_metadata(sel_m, hf_revision)
                version = version_for_revision(sel_m, hf_revision)
                author  = meta["author"]
                tracking.log_evaluation(
                    experiment=tracking.experiment_for(new_row["Benchmark"]),
                    run_name=tracking.make_run_name(new_row["Model"], version or "", author),
                    params={
                        "hf_repo_id":     sel_m,
                        "hf_revision":    hf_revision,
                        "author":         author,
                        "version":        version,
                        "parent_version": meta["parent_version"],
                        **tracking.flatten_config(model_config),
                    },
                    metrics=scores,
                    tags={"author": author, "version": version, "hf_revision": hf_revision},
                )

                st.success(f"Done — {new_row['Model']} on {ds_key} added to scoreboard.")
                st.rerun()

    # The scoreboard reads solely from the MLflow tracking store (the system of
    # record — Neon Postgres via MLFLOW_TRACKING_URI). sb_rows was fetched above.
    if sb_rows:
        show_df = pd.DataFrame(sb_rows).drop(columns=["run_id"], errors="ignore")
        if is_admin:
            st.caption("Check runs on the left, then click **Delete** above to permanently remove them.")
            st.dataframe(
                show_df,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="multi-row",
                key="sb_table",
            )
        else:
            st.dataframe(show_df, use_container_width=True, hide_index=True)
    else:
        st.info("No evaluations recorded yet, or the tracking store is unavailable.")

# ── Admin ─────────────────────────────────────────────────────────────────────

if tab_admin is not None:
    with tab_admin:
        head_l, head_r = st.columns([4, 1], vertical_alignment="center")
        head_l.subheader("Users")
        if head_r.button("Manage Users", use_container_width=True):
            st.session_state.manage_users = not st.session_state.get("manage_users", False)
        st.caption("Everyone registered on the platform. Disabled users are kept but cannot sign in.")

        try:
            users = auth.list_users()
        except Exception as e:
            st.error(f"Could not load users: {e}")
            users = []

        if not users:
            st.info("No users yet.")
        else:
            udf = pd.DataFrame(users)
            udf["is_active"] = udf["is_active"].astype(bool)
            udf["Status"] = udf["is_active"].map(lambda a: "Active" if a else "Disabled")

            m1, m2, m3 = st.columns(3)
            m1.metric("Total users", len(udf))
            m2.metric("Active", int(udf["is_active"].sum()))
            m3.metric("Disabled", int((~udf["is_active"]).sum()))

            show = udf[["username", "email", "role", "Status", "created_at"]].rename(columns={
                "username": "Username", "email": "Email", "role": "Role",
                "created_at": "Registered",
            })
            st.dataframe(show, use_container_width=True, hide_index=True)

            # Management panel — toggled by the "Manage Users" button.
            if st.session_state.get("manage_users", False):
                st.divider()
                st.markdown(
                    "**Manage Users** — disable to revoke access (soft delete; the "
                    "record is kept), or re-enable to restore."
                )
                me = st.session_state.user
                for u in users:
                    c_info, c_status, c_action = st.columns([3, 2, 1], vertical_alignment="center")
                    c_info.markdown(f"**{u['username']}** · {u['email']}")
                    c_status.markdown(
                        f"{u['role']} · {'🟢 Active' if u['is_active'] else '🔴 Disabled'}"
                    )
                    if u["id"] == me["id"]:
                        c_action.caption("(you)")
                    elif u["is_active"]:
                        if c_action.button("Disable", key=f"dis_{u['id']}", use_container_width=True):
                            auth.set_user_active(u["id"], False)
                            st.rerun()
                    else:
                        if c_action.button("Enable", key=f"en_{u['id']}",
                                           type="primary", use_container_width=True):
                            auth.set_user_active(u["id"], True)
                            st.rerun()
