# Define variables for Python interpreter
PYTHON=python

# Define script for model prediction calculations
MODEL_CALCS=1_model_predictions/scripts/model_runner.py

# Define script for hazard calculations
HAZ_CALCS=2_hazard_calcs/scripts/hazard_runner.py

# Define script for fractile calculations & Kumamoto source contributions
FRAC_CALCS=\
	3_fractile_calcs/scripts/fractile_runner.py \
	3_fractile_calcs/scripts/extra_processing_kumamoto_case2.py

# Define scripts for plotting hazard curves
PLOTTING=\
	4_plotting/scripts/plot_curves_runner.py \
	4_plotting/scripts/plot_fdm_comparisons_runner.py \
	4_plotting/scripts/plot_kumamoto_case2_sources.py

# Define script for collecting results into Excel files
EXCEL=5_excel_files/scripts/collecting_runner.py

# Define script for creating report using R markdown
DOCS=6_documentation/scripts/MAIN_REPORT.Rmd
#FIXME: There's a conflict with MikTeX when this Makefile is run in a conda py env
DOCS_WARN=*** The DOCS needs to be run separately, outside conda, don't forget

# Define jobs for make
all: $(MODEL_CALCS) $(HAZ_CALCS) $(FRAC_CALCS) $(PLOTTING) $(EXCEL)
pred: $(MODEL_CALCS)
haz: $(HAZ_CALCS)
fractiles: $(FRAC_CALCS)
plots: $(PLOTTING)
xls: $(EXCEL)
docs: $(DOCS)

# Define targets for make
$(MODEL_CALCS) $(HAZ_CALCS) $(FRAC_CALCS) $(PLOTTING) $(EXCEL):
	cd $(shell dirname $(MAKEFILE_LIST)) && $(PYTHON) $@ && echo "$(DOCS_WARN)"

$(DOCS):
	cd $(shell dirname $(MAKEFILE_LIST)) \
	&& echo "$(DOCS_WARN)" \
	&& Rscript -e "rmarkdown::render('$(DOCS)') "
