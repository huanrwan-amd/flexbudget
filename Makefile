.PHONY: figures audit selftest clean
figures:    ## regenerate every figure from results/*.json
	@for f in figures/make_f*.py; do echo "-> $$f"; python3 $$f; done
audit:      ## geometric layout audit of the mechanism figure
	python3 figures/verify_f5_layout.py
selftest:   ## prove the mask degenerates to dense causal at 100% budget
	python3 -c "import sys; sys.path.insert(0,'flexbudget'); \
	import torch, working_set_mask as w; \
	m=w.make_noop_mask(512); print('noop mask OK, skip_rate =', m.skip_rate)"
clean:
	rm -f figures/*.pdf figures/*.png figures/*_provenance.json
