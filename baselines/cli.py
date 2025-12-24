import typer
from typing import Optional
from .data import read_split, read_meta, map_labels, load_vec
from .cv import run_outer_cv
from .models import REGISTRY
from .scripts.make_cv_splits import make_cv_splits
from .scripts.extract_boss_features import extract_boss_features

app = typer.Typer(add_completion=False)

# ---- helpers ---------------------------------------------------------------
def _is_questionnaire(model_cls) -> bool:
    return getattr(model_cls, "uses_boss_features", True) is False

def cache_features(ids, tags):
    cache = {tag: {} for tag in tags}
    for pid in ids:
        for tag in tags:
            cache[tag][pid] = load_vec(pid, tag)  # returns 1xD csr
    return cache

def build_df(task: str):
    split = read_split()
    meta  = read_meta(cols=("id", "label", "gender"))
    df = split.merge(meta, on="id", how="inner")
    df = map_labels(df, task)   # adds 'y'
    return df

def _parse_csv_list(s: Optional[str]) -> list[str]:
    if s is None:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

# ---- commands --------------------------------------------------------------
@app.command()
def run(
    model: Optional[str] = typer.Option(
        None,
        help="Model key from REGISTRY. Omit to run defaults."
    ),
    task: Optional[str] = typer.Option(
        None,
        help="pd_vs_hc|pd_vs_dd. Omit to run both tasks."
    ),
    tag: Optional[str] = typer.Option(
        None,
        help="rot|acc|all. Movement-only. Omit to run defaults (movement: rot+acc+all; stack: all only). Ignored for questionnaire."
    ),
    use_gpu: bool = typer.Option(False, help="CatBoost only"),
):
    tasks = [task] if task else ["pd_vs_hc", "pd_vs_dd"]

    default_movement_models = ["catboost", "lightgbm", "lr", "mlp", "rf", "svm"]
    default_stack_models = ["stack"]
    default_questionnaire_models = ["quest_catboost", "quest_lr", "quest_svm"]

    # If a single model is specified, just run that (with appropriate defaults)
    if model is not None:
        Model = REGISTRY.get(model)
        if Model is None:
            raise typer.BadParameter(f"unknown model '{model}'")

        is_quest = _is_questionnaire(Model)

        # Questionnaire: no tags, no BOSS cache
        if is_quest:
            for t in tasks:
                df = build_df(t)
                run_outer_cv(df, {}, Model(), task=t, tag="questionnaire", model=model)
            return

        # Stacking: default tag is always 'all'
        if model == "stack":
            tags_to_run = [tag] if tag else ["all"]
            for t in tasks:
                df = build_df(t)
                X_cache = cache_features(df.id.unique(), tags_to_run)
                for tg in tags_to_run:
                    run_outer_cv(df, X_cache, Model(), task=t, tag=tg, model=model)
            return

        # Movement: if no tag, run all tags
        tags_to_run = [tag] if tag else ["rot", "acc", "all"]
        for t in tasks:
            df = build_df(t)
            X_cache = cache_features(df.id.unique(), tags_to_run)
            for tg in tags_to_run:
                model_obj = Model(use_gpu=use_gpu) if ("catboost" in model) else Model()
                run_outer_cv(df, X_cache, model_obj, task=t, tag=tg, model=model)
        return

    # No single model specified: run default sets for movement + stack + questionnaire
    for t in tasks:
        df = build_df(t)

        # ---- movement defaults (rot+acc+all) ----
        mv_tags = [tag] if tag else ["rot", "acc", "all"]
        mv_cache = cache_features(df.id.unique(), mv_tags)

        for m in default_movement_models:
            M = REGISTRY.get(m)
            if M is None:
                raise typer.BadParameter(f"unknown model '{m}'")
            for tg in mv_tags:
                model_obj = M(use_gpu=use_gpu) if ("catboost" in m) else M()
                run_outer_cv(df, mv_cache, model_obj, task=t, tag=tg, model=m)

        # ---- stacking defaults (all only) ----
        st_tags = [tag] if tag else ["all"]  # default all
        st_cache = mv_cache if set(st_tags).issubset(set(mv_tags)) else cache_features(df.id.unique(), st_tags)

        for m in default_stack_models:
            M = REGISTRY.get(m)
            if M is None:
                raise typer.BadParameter(f"unknown model '{m}'")
            for tg in st_tags:
                run_outer_cv(df, st_cache, M(), task=t, tag=tg, model=m)

        # ---- questionnaire defaults ----
        for m in default_questionnaire_models:
            M = REGISTRY.get(m)
            if M is None:
                raise typer.BadParameter(f"unknown model '{m}'")
            run_outer_cv(df, {}, M(), task=t, tag="questionnaire", model=m)


@app.command()
def make_splits():
    make_cv_splits()

@app.command()
def extract_boss():
    extract_boss_features()

def main():
    app()

if __name__ == "__main__":
    main()
