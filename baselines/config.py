from pathlib import Path
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    root: Path = Field(default=Path.cwd(), env="ROOT")  # picks os.environ["ROOT"] if set
    seed: int = 42

    @property
    def feat_dir(self):  return self.root / "baselines" / "boss" / "features_bin"
    @property
    def split_csv(self): return self.root / "cv_splits.csv"
    @property
    def meta_csv(self):  return self.root / "preprocessed" / "file_list.csv"
    @property
    def bins_dir(self):  return self.root / "preprocessed" / "movement"  # for .bin files
    @property
    def questionnaire_dir(self): return self.root / "questionnaire"

cfg = Settings()
