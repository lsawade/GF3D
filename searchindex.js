Search.setIndex({"docnames": ["examples/client/client_usage_demo", "examples/client/index", "examples/extraction/database/index", "examples/extraction/database/run_make_subset", "examples/extraction/database/run_single_seismogram_db", "examples/extraction/database/run_station_section_db", "examples/extraction/database/sg_execution_times", "examples/extraction/subset/index", "examples/extraction/subset/run_aligned_stations", "examples/extraction/subset/run_single_seismogram", "examples/extraction/subset/run_station_section", "examples/extraction/subset/sg_execution_times", "examples/generation/index", "examples/generation/simulation", "index", "parts/api/index", "parts/gf-extraction/index", "parts/gf-generation/index", "parts/gf-generation/installation/adios", "parts/gf-generation/installation/creating-environment", "parts/gf-generation/installation/h5py", "parts/gf-generation/installation/hdf5", "parts/gf-generation/installation/index", "parts/gf-generation/installation/mpi4py"], "filenames": ["examples/client/client_usage_demo.rst", "examples/client/index.rst", "examples/extraction/database/index.rst", "examples/extraction/database/run_make_subset.rst", "examples/extraction/database/run_single_seismogram_db.rst", "examples/extraction/database/run_station_section_db.rst", "examples/extraction/database/sg_execution_times.rst", "examples/extraction/subset/index.rst", "examples/extraction/subset/run_aligned_stations.rst", "examples/extraction/subset/run_single_seismogram.rst", "examples/extraction/subset/run_station_section.rst", "examples/extraction/subset/sg_execution_times.rst", "examples/generation/index.rst", "examples/generation/simulation.rst", "index.rst", "parts/api/index.rst", "parts/gf-extraction/index.rst", "parts/gf-generation/index.rst", "parts/gf-generation/installation/adios.rst", "parts/gf-generation/installation/creating-environment.rst", "parts/gf-generation/installation/h5py.rst", "parts/gf-generation/installation/hdf5.rst", "parts/gf-generation/installation/index.rst", "parts/gf-generation/installation/mpi4py.rst"], "titles": ["First example of client usage to create subset.", "Using the client to retrieve database subsets and seimograms", "Database Extraction Tutorials", "Create subset from database files", "Single Seismogram", "Station Section", "Computation times", "Subset Extraction Tutorials", "Aligned Station Section", "Single Seismogram", "Station Section", "Computation times", "Gallery", "Simulation Class Demo", "GF3D Documentation", "API Documentation", "GF Extraction", "GF Generation", "<cite>adios2</cite>", "Creating an environment", "<cite>h5py</cite> Installation", "<cite>HDF5</cite> Installation", "Custom Installation for Database Creation", "<cite>mpi4py</cite>"], "terms": {"go": [0, 3, 4, 5, 8, 9, 10, 13, 14, 16], "end": [0, 3, 4, 5, 8, 9, 10, 13], "download": [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13], "full": [0, 3, 4, 5, 8, 9, 10, 13, 15], "code": [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 16], "sinc": 0, "databas": [0, 4, 5, 6, 9, 13, 15, 16, 17], "veri": [0, 15], "larg": 0, "we": [0, 1, 4, 5, 8, 9, 10, 16, 17], "almost": 0, "never": 0, "want": [0, 14, 15, 16], "entir": 0, "thing": 0, "do": [0, 15], "some": [0, 1, 2, 7, 16, 17], "cmt": [0, 3, 4, 5, 8, 9, 10, 15, 16], "invers": [0, 15], "so": [0, 4, 5, 15, 16, 17], "better": 0, "workflow": 0, "i": [0, 4, 5, 8, 9, 10, 14, 15, 16, 22], "follow": [0, 15, 16], "region": [0, 15], "server": [0, 16], "where": [0, 15], "locat": [0, 15, 16], "file": [0, 2, 4, 5, 6, 8, 9, 10, 11, 13, 15, 16, 18, 22], "from": [0, 2, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 22], "load": [0, 3, 15, 16], "he": 0, "gfmanag": [0, 3, 4, 5, 8, 9, 10, 15, 16], "read_gf": 0, "extract": [0, 14], "green": [0, 2, 7, 8, 22], "function": [0, 2, 7, 8, 22], "note": [0, 4, 5, 9, 15], "thi": [0, 13, 14, 15, 16, 17, 22], "cannot": 0, "run": [0, 3, 4, 5, 8, 9, 10, 13, 15, 16], "galleri": [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 13, 16, 17], "requir": [0, 15, 22], "queri": [0, 4, 9, 14, 15, 16], "As": [0, 16], "result": 0, "all": [0, 1, 2, 7, 12, 15, 16, 17, 22], "output": [0, 15, 16, 18], "ar": [0, 4, 5, 9, 15, 16, 17, 22], "hand": 0, "written": [0, 15], "mai": 0, "contain": [0, 15, 16], "error": 0, "data": [0, 3, 4, 5, 8, 9, 10, 15], "single_element_read": [0, 3, 4, 5, 8, 9, 10], "db": [0, 3, 4, 5, 15], "modul": [0, 14], "import": [0, 3, 4, 5, 8, 9, 10, 15, 16, 17], "o": 0, "subprocess": 0, "check_cal": 0, "gf3d": [0, 1, 3, 4, 5, 8, 9, 10, 14, 15, 16, 17, 19, 22], "seismogram": [0, 2, 3, 5, 6, 7, 10, 11, 16, 17], "sourc": [0, 1, 2, 3, 4, 5, 7, 9, 10, 12, 13], "cmtsolut": [0, 3, 4, 5, 8, 9, 10, 15, 16], "The": [0, 4, 5, 8, 9, 10, 15, 16], "automat": [0, 16], "know": [0, 16], "about": [0, 5, 22], "given": [0, 15], "e": [0, 5], "thei": [0, 4], "hard": [0, 16], "gfcl": 0, "princeton": [0, 16], "With": 0, "initi": [0, 15], "can": [0, 3, 4, 5, 9, 15, 16, 17, 22], "dictionari": 0, "gener": [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 16], "paramet": [0, 5, 15, 16], "info": [0, 1, 15, 16], "get_info": 0, "print": [0, 3, 4, 9, 13, 16], "topographi": [0, 15], "true": [0, 4, 15], "ellipt": [0, 15], "nx_topo": 0, "5400": 0, "ny_topo": 0, "2700": 0, "res_topo": 0, "4": [0, 4, 8, 9, 16], "0": [0, 3, 4, 5, 6, 8, 9, 10, 11, 13, 15, 16], "nspl": 0, "628": 0, "nspec": 0, "1": [0, 3, 4, 5, 8, 9, 10, 15, 16], "nglob": 0, "125": [0, 3], "ngllx": 0, "5": [0, 3, 4, 5, 15, 16], "nglly": 0, "ngllz": 0, "dt": [0, 15], "900000035000001": 0, "tc": [0, 15], "200": 0, "nstep": [0, 3, 15], "776": [0, 3], "factor": [0, 15], "1e": 0, "17": [0, 4, 9, 16], "hdur": [0, 15, 16], "700000005": 0, "use_buffer_el": 0, "fals": [0, 8, 15], "also": [0, 15], "station": [0, 1, 2, 4, 6, 7, 9, 11, 14, 15, 16], "avail": [0, 1], "stations_avail": 0, "iu": [0, 3, 5], "hrv": [0, 3, 5], "anmo": [0, 3, 5], "ii": [0, 3, 4, 5, 9], "bfo": [0, 3, 4, 5, 9, 15], "now": [0, 4, 9, 15], "inform": 0, "quit": [0, 16], "easili": 0, "well": [0, 16, 18], "set": [0, 4, 8, 9, 15, 22], "latitud": [0, 3, 4, 5, 8, 9, 15, 16], "31": [0, 4, 9, 16], "1300": [0, 4, 9, 16], "longitud": [0, 3, 4, 5, 8, 9, 15, 16], "72": [0, 4, 9, 16], "0900": [0, 4, 9, 16], "depth_in_km": [0, 16], "3500": [0, 4, 9, 16], "radius_in_km": [0, 16], "28": [0, 4, 9, 16], "onli": [0, 5, 14, 15, 16], "chose": 0, "becaus": 0, "get": [0, 4, 5, 9, 15, 16, 17], "singl": [0, 2, 6, 7, 11, 15], "element": [0, 4, 5, 15, 16], "make": [0, 15, 16, 17], "get_subset": 0, "firstqueri": 0, "h5": [0, 3, 4, 5, 8, 9, 10, 16], "It": [0, 16], "ll": 0, "take": [0, 15, 16], "minut": [0, 3, 4, 5, 8, 9, 10, 13, 15, 16], "dataset": 0, "should": [0, 15, 16], "start": [0, 16], "progress": 0, "bar": 0, "show": [0, 1, 2, 4, 5, 7, 8, 9, 10, 13, 16], "let": 0, "": [0, 3, 4, 8, 15, 16, 17], "python": [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 16, 19, 23], "gfm": [0, 3, 4, 5, 15], "read": [0, 3, 4, 5, 8, 9, 10, 15, 18], "seismo": 0, "rp": [0, 3, 4, 5, 8, 9, 10], "get_seismogram": [0, 3, 4, 5, 8, 9, 10, 15], "9": [0, 3, 4, 8, 9], "trace": [0, 3, 4, 5, 8, 9, 10, 15, 16], "stream": [0, 3, 4, 5, 8, 10, 15], "mxn": [0, 3], "2015": [0, 3, 4, 9], "09": [0, 3], "16t22": [0, 3], "51": [0, 3], "12": [0, 3], "900000z": [0, 3], "16t23": [0, 3], "54": [0, 3, 4, 9], "30": [0, 3], "400027z": [0, 3], "2": [0, 3, 4, 5, 8, 9, 15, 16], "hz": [0, 3], "sampl": [0, 3, 15, 16], "mxe": [0, 3], "mxz": [0, 3], "would": [0, 15, 16], "still": [0, 17], "made": 0, "fortranqueri": 0, "For": [0, 8, 16], "line": 0, "below": [0, 16, 22], "work": [0, 17, 22], "you": [0, 3, 4, 5, 13, 14, 15, 16, 17, 18, 21], "need": [0, 14, 15, 16, 17, 18, 21], "path": [0, 18], "build": [0, 20], "export": [0, 20, 23], "absolut": 0, "gf3df": 0, "bin": 0, "cmd": 0, "sac": [0, 4, 5, 8, 9, 10], "sdp": 0, "subsetfil": [0, 16], "cmtfile": 0, "outdir": [0, 16], "itypsokern": [0, 16], "3": [0, 3, 4, 5, 9, 15, 16, 19], "an": [0, 7, 8, 14, 15, 17, 22], "directori": [0, 8, 10, 15, 16, 22], "doe": [0, 15], "itself": 0, "exist": [0, 14], "makedir": 0, "f": [0, 8, 14], "shell": 0, "total": [0, 3, 4, 5, 6, 8, 9, 10, 11, 13], "time": [0, 3, 4, 5, 8, 9, 10, 13, 15, 16], "script": [0, 3, 4, 5, 8, 9, 10, 13], "000": [0, 13], "second": [0, 3, 4, 5, 8, 9, 10, 13, 16], "client_usage_demo": 0, "py": [0, 3, 4, 5, 6, 8, 9, 10, 11, 13], "jupyt": [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13], "notebook": [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13], "ipynb": [0, 3, 4, 5, 8, 9, 10, 13], "sphinx": [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13], "here": [1, 4, 5, 16], "exampl": [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13], "how": [1, 13, 15, 16], "gf3dclient": 1, "more": 1, "first": [1, 4, 5, 16], "usag": [1, 16], "creat": [1, 2, 6, 12, 14, 15, 17, 22], "client_python": 1, "zip": [1, 2, 7, 12], "client_jupyt": 1, "visual": [2, 7], "3d": [2, 7], "raw": [2, 4, 5, 8, 9, 10], "subset": [2, 4, 5, 6, 8, 9, 10, 16], "section": [2, 6, 7, 11], "database_python": 2, "database_jupyt": 2, "extern": [3, 4, 5, 8, 9, 10], "glob": [3, 5], "intern": [3, 4, 5, 8, 9, 10], "event": [3, 4, 5, 9, 15], "load_header_vari": [3, 4, 5, 15], "get_el": [3, 4, 5, 15], "depth": [3, 4, 5, 8, 9, 15, 16], "dist_in_km": 3, "100": [3, 4, 5, 8], "ngll": [3, 4, 5, 16], "hdf5": [3, 4, 5, 14, 17, 20, 22], "mode": [3, 4, 5], "r": [3, 4, 5, 16], "write_subset": [3, 15], "temp_subset": 3, "durat": [3, 4, 5, 8, 9, 10, 15], "3600": [3, 4, 10], "shape": 3, "gfsub": [3, 8, 9, 10], "final": [3, 4, 5], "s3": 3, "244": [3, 6], "run_make_subset": [3, 6], "tutori": [4, 5, 8, 9, 10, 16], "over": [4, 5, 8, 9, 10, 16], "three": 4, "compon": [4, 5], "one": [4, 5, 8, 9, 10, 15, 22], "includ": [4, 5, 8, 9, 10, 16, 22], "github": [4, 5, 9, 14, 19], "repo": [4, 5], "At": [4, 5, 8, 9, 10], "plot": [4, 5, 9, 10, 16], "us": [4, 5, 8, 9, 10, 13, 15, 16, 22], "built": [4, 5, 8, 9, 10], "tool": [4, 5, 8, 9, 10], "matplotlib": [4, 5, 8, 9, 10], "pyplot": [4, 5, 8, 9, 10], "plt": [4, 5, 8, 9, 10], "obspi": [4, 5, 8, 9, 10, 15], "read_inventori": [4, 5, 8, 9, 10], "process": [4, 5, 9, 10, 15, 22], "process_stream": [4, 5, 8, 9, 10], "plotseismogram": [4, 9], "inv": [4, 5, 8, 9, 10], "xml": [4, 5, 8, 9, 10], "pde": [4, 9, 15], "16": [4, 9], "22": [4, 9], "32": [4, 9], "90": [4, 9], "5700": [4, 9], "71": [4, 9], "6700": [4, 9], "8": [4, 9, 15], "NEAR": [4, 9], "coast": [4, 9], "OF": [4, 9], "central": [4, 9], "ch": [4, 9], "name": [4, 9], "201509162254a": [4, 9], "shift": [4, 9, 16], "49": [4, 9], "9800": [4, 9], "half": [4, 9, 10, 15], "33": [4, 9, 10], "4000": [4, 9, 10], "mrr": [4, 9, 15, 16], "950000e": [4, 9], "mtt": [4, 9, 15, 16], "360000e": [4, 9], "26": [4, 9], "mpp": [4, 9, 15, 16], "910000e": [4, 9], "mrt": [4, 9, 15, 16], "7": [4, 9], "420000e": [4, 9], "27": [4, 9], "mrp": [4, 9, 15, 16], "480000e": [4, 9], "mtp": [4, 9, 15, 16], "ha": [4, 9], "been": [4, 9, 15], "iri": [4, 9], "previous": [4, 9], "synthet": [4, 5, 8, 9, 10, 16], "correspond": [4, 9], "observ": [4, 5, 8, 9, 10], "ones": [4, 9], "filenam": 4, "have": [4, 5, 15], "just": [4, 5], "storag": [4, 5], "reason": [4, 5], "Then": [4, 9, 15], "select": [4, 8, 9, 15], "question": [4, 9], "both": [4, 9], "band": [4, 9, 16], "pass": [4, 9, 15, 16], "40": [4, 8, 9], "300": [4, 8, 9], "network": [4, 5, 9, 15], "ob": [4, 5, 8, 9, 10], "3300": [4, 5, 8, 9, 10], "syn": [4, 5, 8, 9, 10], "limit": [4, 5, 9, 10], "starttimeoffset": [4, 9], "endtimeoffset": [4, 9], "stat": [4, 5, 9, 10], "starttim": [4, 5, 9, 10, 15, 16], "nooffset": 4, "lw": [4, 5, 8, 10], "25": 4, "offset": [4, 9], "clariti": [4, 9], "easi": 4, "405": [4, 6], "run_single_seismogram_db": [4, 6], "waveform": [5, 8, 9, 10], "select_pair": [5, 8, 10], "plotsect": [5, 10], "which": [5, 16, 17, 23], "mean": [5, 15], "other": [5, 16], "pob": [5, 8, 10], "psyn": [5, 8, 10], "cant": 5, "find": [5, 15], "hope": 5, "n": [5, 14, 19], "z": [5, 8, 10], "kapi": 5, "dgar": 5, "dav": 5, "ctao": 5, "kdak": 5, "aak": 5, "majo": 5, "sba": 5, "kowa": 5, "borg": 5, "yak": 5, "xma": 5, "snzo": 5, "don": 5, "t": [5, 15, 16], "worri": 5, "endtim": [5, 10], "comp": [5, 8, 10], "75": [5, 10], "630": [5, 6], "run_station_section_db": [5, 6], "00": [6, 11], "02": [6, 11], "278": 6, "execut": [6, 11], "examples_extraction_databas": 6, "01": [6, 11], "mb": [6, 11, 15], "align": [7, 11], "subset_python": 7, "subset_jupyt": 7, "util": 8, "set_default_color": 8, "section_align": 8, "plotsection_align": 8, "get_azimuth_distance_traveltim": 8, "filter_st": 8, "single_el": [8, 9, 10], "window": 8, "thee": 8, "windowp": 8, "250": 8, "pwave": 8, "traveltime_window": 8, "p": 8, "sob": 8, "ssyn": 8, "match": 8, "_i": 8, "fig": 8, "figur": 8, "figsiz": 8, "6": [8, 15, 16], "arriv": 8, "around": [8, 15], "ak135": 8, "ax": 8, "subplot": 8, "labelright": 8, "labelleft": 8, "titl": 8, "cmt_time": [8, 15], "ctime": 8, "loc": 8, "2f": 8, "dg": 8, "1f": 8, "km": [8, 15, 16], "bp": 8, "suptitl": 8, "fontsiz": 8, "small": [8, 15], "adjust": 8, "subplots_adjust": 8, "left": 8, "right": [8, 17], "85": 8, "bottom": [8, 16], "top": 8, "wspace": 8, "104": [8, 11], "run_aligned_st": [8, 11], "repositori": 9, "respons": 9, "500": 9, "1000": 9, "374": [9, 11], "run_single_seismogram": [9, 11], "numpi": 10, "np": [10, 15], "signal": 10, "filter": [10, 15], "butter_low_two_pass_filt": 10, "966": [10, 11], "run_station_sect": [10, 11], "05": 11, "444": 11, "examples_extraction_subset": 11, "To": 12, "simul": [12, 15], "class": [12, 15, 16], "demo": 12, "generation_python": 12, "generation_jupyt": 12, "setup": [13, 15, 16], "specfem3d_glob": [13, 14, 15, 17, 22], "subsequ": [13, 16], "under": 13, "construct": 13, "librari": 14, "specif": [14, 15, 16], "basi": 14, "environ": [14, 16, 17, 22], "yml": 14, "current": 14, "wai": [14, 15], "git": [14, 19], "clone": [14, 19], "http": [14, 19], "com": [14, 19], "lsawad": [14, 19, 20], "cd": [14, 19], "conda": [14, 18, 19, 22], "env": [14, 19], "packag": [14, 18, 20, 22], "manag": [14, 15], "come": 14, "soon": 14, "pypi": 14, "pip": [14, 20, 23], "c": 14, "forg": 14, "interest": 14, "manual": 14, "parallel": [14, 21, 22], "adio": [14, 18, 22], "see": [14, 16], "index": [14, 16], "search": 14, "page": [14, 16, 17], "specfemdir": 15, "stationdir": 15, "str": 15, "none": 15, "station_latitud": 15, "float": 15, "station_longitud": 15, "station_buri": 15, "target_fil": 15, "target_latitud": 15, "option": [15, 16], "union": 15, "ndarrai": 15, "iter": 15, "target_longitud": 15, "target_depth": 15, "par_fil": 15, "element_buff": 15, "int": 15, "force_factor": 15, "100000000000000": 15, "t0": 15, "duration_in_min": 15, "20": 15, "subsampl": [15, 17], "bool": 15, "ndt": 15, "lpfilter": 15, "butter": 15, "forward_test": 15, "forwardoutdir": 15, "broadcast_mesh_model": 15, "simultaneous_run": 15, "cmtsolutionfil": 15, "overwrit": 15, "specfem": [15, 18], "_type_": 15, "base": 15, "gf": [15, 16], "default": 15, "burial": 15, "target": 15, "gf_locat": 15, "tp": 15, "relev": 15, "up": [15, 22], "mani": 15, "buffer": 15, "tag": 15, "reciproc": 15, "forc": 15, "1e14": 15, "center": 15, "centroid": [15, 16], "number": [15, 16], "timestep": 15, "hardset": 15, "flag": 15, "turn": 15, "off": 15, "dure": 15, "request": 15, "new": 15, "rate": [15, 16], "type": [15, 16], "low": 15, "check": 15, "whether": 15, "perform": 15, "test": 15, "separ": 15, "comput": [15, 16], "forward": 15, "auto": 15, "necessari": 15, "anymor": 15, "certain": 15, "direcori": 15, "same": 15, "normal": [15, 16], "That": 15, "nex_": 15, "nproc_": 15, "nchunk": 15, "model": [15, 22], "rotat": 15, "graviti": 15, "attenu": 15, "use_adio": 15, "befor": 15, "mesher": 15, "If": [15, 16], "provid": 15, "downsampl": 15, "valu": [15, 16], "approxim": 15, "accur": 15, "period": [15, 16], "what": 15, "edit": 15, "constant": 15, "h": 15, "revers": 15, "mesh": 15, "write": [15, 16], "forcesolut": 15, "correct": 15, "length": 15, "nt": 15, "postprocess": 15, "combin": 15, "sgt": [15, 20], "each": 15, "method": 15, "check_input": 15, "setup_forward": 15, "update_forces_and_st": 15, "write_forc": 15, "write_gf_loc": 15, "write_par_fil": 15, "write_stf": 15, "create_specfem": 15, "after": 15, "actual": 15, "get_timestep_period": 15, "header": 15, "fix": 15, "smaller": 15, "than": 15, "timesstep": 15, "unchang": 15, "self": [15, 16], "remain": 15, "read_gf_loc": 15, "must": 15, "format": [15, 22], "structur": [15, 16], "update_const": 15, "updat": 15, "write_cmt": 15, "otherwis": 15, "doesn": 15, "matter": 15, "write_st": 15, "list": 15, "expect": 15, "your": [15, 16, 17, 22], "complet": 15, "consist": 15, "exact": 15, "amount": 15, "etc": 15, "goal": 15, "k": 15, "closest": 15, "10": [15, 18, 19], "due": 15, "standard": [15, 16], "adjac": 15, "strain": 15, "restructur": 15, "ibool": 15, "xyz": 15, "retreiv": 15, "fast": 15, "much": 15, "kdtree": 15, "global": 15, "purpos": 15, "return": 15, "attribut": 15, "xadj": 15, "get_mesh_loc": 15, "get_mt_frechet": 15, "get_frechet": 15, "rtype": 15, "finit": 15, "differ": 15, "10m": 15, "perturb": 15, "epsilon": 15, "arrai": 15, "load_scalar_header_paramet": 15, "scalar": 15, "load_subset_header_onli": 15, "outfil": 15, "fortran": [15, 16], "origin_tim": 15, "core": 15, "utcdatetim": 15, "2000": 15, "pde_lat": 15, "pde_lon": 15, "pde_depth": 15, "m": [15, 23], "region_tag": 15, "eventnam": 15, "time_shift": 15, "repres": 15, "classic": 15, "implement": [15, 16], "origin": [15, 16], "deg": [15, 16], "bodi": 15, "wave": 15, "magnitud": 15, "surfac": 15, "id": 15, "timeshift": 15, "moment": [15, 16], "tensor": [15, 16], "dyn": [15, 16], "m0": 15, "nm": 15, "mw": 15, "m_w": 15, "utc": 15, "fulltensor": 15, "3x3": 15, "from_ev": 15, "read_quakeml": 15, "same_eventid": 15, "48": 15, "3319": 15, "3311": 15, "stf": 15, "forcefactor": 15, "vector_": 15, "vector_n": 15, "vector_z_up": 15, "force_no": 15, "goe": 16, "advanc": 16, "case": 16, "two": 16, "main": 16, "memori": 16, "extens": 16, "demonstr": 16, "host": 16, "sure": 16, "activ": 16, "instal": [16, 17, 18, 23], "like": 16, "subcommand": 16, "subsubcommand": 16, "arg1": 16, "arg2": 16, "mayb": 16, "help": [16, 17], "arg": 16, "messag": 16, "exit": 16, "sub": 16, "These": 16, "indic": 16, "call": 16, "NOT": 16, "databasenam": 16, "subsetfilenam": 16, "neg": 16, "boolean": 16, "greet": 16, "integ": 16, "gll": 16, "point": 16, "netsta": 16, "text": 16, "subselect": 16, "A": [16, 22], "look": 16, "testqueri": 16, "enter": 16, "distinguish": 16, "henc": 16, "click": 16, "tell": 16, "argument": 16, "fact": 16, "detail": 16, "pleas": 16, "visit": 16, "poster": 16, "abov": [16, 22], "immens": 16, "power": 16, "when": 16, "signatur": 16, "yyyi": 16, "mm": 16, "dd": 16, "hh": 16, "ss": 16, "ssss": 16, "tshift": 16, "tfir": 16, "spsam": 16, "it1smp": 16, "it2smp": 16, "pminswrai": 16, "pmaxswrai": 16, "long": 16, "everi": 16, "essenti": 16, "subset_fil": 16, "year": 16, "month": 16, "dai": 16, "hour": 16, "cm": 16, "half_dur": 16, "w": 16, "per": 16, "interpol": 16, "last": 16, "npt": 16, "minimum": 16, "maximum": 16, "kernel": 16, "desir": 16, "most": [16, 22], "explanatori": 16, "descript": 16, "subroutin": 16, "synt": 16, "store": [16, 22], "par": 16, "client": 16, "retriev": 16, "seimogram": 16, "underneat": 17, "hopefulli": 17, "throughout": 17, "cover": 17, "topic": 17, "alias": 17, "precis": 17, "decis": 17, "custom": 17, "creation": 17, "mpi4pi": [17, 22], "h5py": [17, 22], "adios2": [17, 22], "compil": [18, 21, 22], "develop": 18, "dir": [18, 20], "lib": 18, "python3": 18, "site": 18, "summit": 20, "hdf5_dir": 20, "gpf": 20, "alpin": 20, "geo111": 20, "scratch": 20, "specfemmag": 20, "travers": 20, "cc": 20, "mpicc": [20, 23], "hdf5_mpi": 20, "ON": 20, "binari": 20, "cach": 20, "support": 21, "There": 22, "post": 22, "version": 22, "wa": 22, "paral": 22, "step": 22, "explain": 22, "everyth": 22, "fit": 22, "especi": 22, "sometim": 22, "annoi": 22, "talk": 22, "variabl": 22}, "objects": {"": [[14, 0, 0, "-", "gf-index"], [15, 0, 0, "-", "gf3d"]], "gf3d": [[16, 0, 0, "-", "seismograms"], [17, 0, 0, "-", "simulation"]], "gf3d.seismograms": [[15, 1, 1, "", "GFManager"]], "gf3d.seismograms.GFManager": [[15, 2, 1, "", "get_frechet"], [15, 2, 1, "", "load"], [15, 2, 1, "", "load_scalar_header_parameters"], [15, 2, 1, "", "load_subset_header_only"], [15, 2, 1, "", "write_subset"]], "gf3d.simulation": [[15, 1, 1, "", "Simulation"]], "gf3d.simulation.Simulation": [[15, 2, 1, "", "create_specfem"], [15, 2, 1, "", "get_timestep_period"], [15, 2, 1, "", "read_GF_LOCATIONS"], [15, 2, 1, "", "setup"], [15, 2, 1, "", "update_constants"], [15, 2, 1, "", "write_CMT"], [15, 2, 1, "", "write_STATIONS"]], "gf3d.source": [[15, 1, 1, "", "CMTSOLUTION"], [15, 1, 1, "", "FORCESOLUTION"]]}, "objtypes": {"0": "py:module", "1": "py:class", "2": "py:method"}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "class", "Python class"], "2": ["py", "method", "Python method"]}, "titleterms": {"first": 0, "exampl": [0, 16], "client": [0, 1], "usag": 0, "creat": [0, 3, 19], "subset": [0, 1, 3, 7], "gf3dclient": 0, "retriev": [0, 1], "us": [0, 1], "fortran": 0, "api": [0, 15], "databas": [1, 2, 3, 14, 22], "seimogram": 1, "extract": [2, 7, 15, 16], "tutori": [2, 7], "from": 3, "file": 3, "singl": [4, 9], "seismogram": [4, 8, 9], "load": [4, 5, 8, 9, 10], "all": [4, 5, 8, 9, 10], "modul": [4, 5, 8, 9, 10], "station": [5, 8, 10], "section": [5, 8, 10], "comput": [6, 11], "time": [6, 11], "align": 8, "get": 8, "sourc": [8, 15], "process": 8, "plot": 8, "galleri": 12, "simul": 13, "class": 13, "demo": 13, "green": [14, 15, 16, 17], "function": [14, 15, 16, 17], "3d": [14, 15, 16, 17], "gener": [14, 15, 17], "extractor": 14, "quick": 14, "instal": [14, 20, 21, 22], "custom": [14, 22], "creation": [14, 22], "indic": 14, "tabl": 14, "document": 15, "command": 16, "line": 16, "interfac": 16, "cli": 16, "old": 16, "tool": 16, "adios2": 18, "an": 19, "environ": 19, "h5py": 20, "hdf5": 21, "mpi4pi": 23}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.viewcode": 1, "sphinx.ext.intersphinx": 1, "sphinx": 57}, "alltitles": {"First example of client usage to create subset.": [[0, "first-example-of-client-usage-to-create-subset"]], "GF3DClient": [[0, "gf3dclient"]], "Retrieving a subset using the client": [[0, "retrieving-a-subset-using-the-client"]], "Using the Fortran API": [[0, "using-the-fortran-api"]], "Using the client to retrieve database subsets and seimograms": [[1, "using-the-client-to-retrieve-database-subsets-and-seimograms"]], "Database Extraction Tutorials": [[2, "database-extraction-tutorials"]], "Create subset from database files": [[3, "create-subset-from-database-files"]], "Single Seismogram": [[4, "single-seismogram"], [9, "single-seismogram"]], "Loading all modules": [[4, "loading-all-modules"], [5, "loading-all-modules"], [8, "loading-all-modules"], [9, "loading-all-modules"], [10, "loading-all-modules"]], "Station Section": [[5, "station-section"], [10, "station-section"]], "Computation times": [[6, "computation-times"], [11, "computation-times"]], "Subset Extraction Tutorials": [[7, "subset-extraction-tutorials"]], "Aligned Station Section": [[8, "aligned-station-section"]], "Get Sources and Seismograms": [[8, "get-sources-and-seismograms"]], "Process": [[8, "process"]], "Plot section": [[8, "plot-section"]], "Gallery": [[12, "gallery"]], "Simulation Class Demo": [[13, "simulation-class-demo"]], "Green Functions 3D Generator & Extractor": [[14, "green-functions-3d-generator-extractor"]], "Quick-Install": [[14, "quick-install"]], "Custom Installation for Database Creation": [[14, "custom-installation-for-database-creation"], [22, "custom-installation-for-database-creation"]], "Indices and tables": [[14, "indices-and-tables"]], "API Documentation": [[15, "api-documentation"]], "3D Green Function Generation": [[15, "d-green-function-generation"], [17, "d-green-function-generation"]], "3D Green Function Extraction": [[15, "d-green-function-extraction"], [16, "d-green-function-extraction"]], "Sources": [[15, "sources"]], "Command Line Interface (CLI)": [[16, "command-line-interface-cli"]], "OLD command line tool": [[16, "old-command-line-tool"]], "Examples": [[16, "examples"]], "adios2": [[18, "adios2"]], "Creating an environment": [[19, "creating-an-environment"]], "h5py Installation": [[20, "h5py-installation"]], "HDF5 Installation": [[21, "hdf5-installation"]], "mpi4py": [[23, "mpi4py"]]}, "indexentries": {"gf-index": [[14, "module-gf-index"]], "module": [[14, "module-gf-index"], [15, "module-gf3d"], [16, "module-gf3d.seismograms"], [17, "module-gf3d.simulation"]], "cmtsolution (class in gf3d.source)": [[15, "gf3d.source.CMTSOLUTION"]], "forcesolution (class in gf3d.source)": [[15, "gf3d.source.FORCESOLUTION"]], "gfmanager (class in gf3d.seismograms)": [[15, "gf3d.seismograms.GFManager"]], "simulation (class in gf3d.simulation)": [[15, "gf3d.simulation.Simulation"]], "create_specfem() (simulation method)": [[15, "gf3d.simulation.Simulation.create_specfem"]], "get_frechet() (gfmanager method)": [[15, "gf3d.seismograms.GFManager.get_frechet"]], "get_timestep_period() (simulation method)": [[15, "gf3d.simulation.Simulation.get_timestep_period"]], "gf3d": [[15, "module-gf3d"]], "load() (gfmanager method)": [[15, "gf3d.seismograms.GFManager.load"]], "load_scalar_header_parameters() (gfmanager method)": [[15, "gf3d.seismograms.GFManager.load_scalar_header_parameters"]], "load_subset_header_only() (gfmanager method)": [[15, "gf3d.seismograms.GFManager.load_subset_header_only"]], "read_gf_locations() (simulation method)": [[15, "gf3d.simulation.Simulation.read_GF_LOCATIONS"]], "setup() (simulation method)": [[15, "gf3d.simulation.Simulation.setup"]], "update_constants() (simulation method)": [[15, "gf3d.simulation.Simulation.update_constants"]], "write_cmt() (simulation method)": [[15, "gf3d.simulation.Simulation.write_CMT"]], "write_stations() (simulation method)": [[15, "gf3d.simulation.Simulation.write_STATIONS"]], "write_subset() (gfmanager method)": [[15, "gf3d.seismograms.GFManager.write_subset"]], "gf3d.seismograms": [[16, "module-gf3d.seismograms"]], "gf3d.simulation": [[17, "module-gf3d.simulation"]]}})