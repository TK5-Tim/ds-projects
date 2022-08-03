"""Scraper for https://fotmob.com/"""

import itertools
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from soccerdata import FBref

from ._common import BaseRequestsReader, make_game_id, standardize_colnames
from ._config import DATA_DIR, NOCACHE, NOSTORE, TEAMNAME_REPLACEMENTS, logger

FOTMOB_DATADIR = DATA_DIR / "fotmob"
FOTMOB_API = "https://www.fotmob.com/api"

class Fotmob(BaseRequestsReader):
    """Provides pd.DataFrames from data at https://www.fotmob.com.

    Data will be downloaded as necessary and cached locally in 
    ``~/soccerdata/data/fotmob``.

    Parameters
    ----------
    
    """

    def __init__(
        self,
        leagues: Optional[Union[str, List[str]]] = None,
        seasons: Optional[Union[str, List[str]]] = None,
        proxy: Optional[
            Union[str, Dict[str, str], List[Dict[str, str]], Callable[[], Dict[str, str]]]
        ] = None,
        no_cache: bool = NOCACHE,
        no_store: bool = NOSTORE,
        data_dir: Path = FOTMOB_DATADIR,
    ):
        super().__init__(
            leagues=leagues,
            proxy=proxy,
            no_cache=no_cache,
            no_store=no_store,
            data_dir=data_dir,
        )
        self.rate_limit = 3
        self.seasons = seasons # type: ignore
    def read_leagues(self) -> pd.DataFrame:
        """Retrieve selected leagues from the datasource.
        
        Returns
        -------
        pd.DataFrame
        """
        url = "https://www.fotmob.com/api/allLeagues"
        fielpath = self.data_dir / "leagues.json"
        reader = self.get(url, fielpath)


        return self.read_data("leagues")
